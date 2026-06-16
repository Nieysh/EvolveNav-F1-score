import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.utils import logging
from .ops import pad_tensors_wgrad, gen_seq_masks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from .image_embedding import ImageEmbeddings
from .modified_lm import ModifiedOPTForCasualLM, ModifiedLlamaForCausalLM, TrieLogitsProcessor
from typing import Dict, List, Any
import os
import numpy as np
import math

logging.set_verbosity_error()


def init_vis_config(args, config):
    cfg_name = os.path.join(args.data_dir, 'bert-large-uncased')
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    vis_config.num_pano_layers = config.num_pano_layers
    vis_config.precision = args.precision
    vis_config.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.type_vocab_size = 3
    return vis_config


class NavModel(nn.Module):
    def __init__(self, args, logger, model_config):
        super().__init__()
        self.args = args
        config = init_vis_config(args, model_config)
        self.config = config

        # Large Language Model
        if args.resume_from_checkpoint is not None or args.from_scratch:
            logger.info("Initialize the model from config.")
            model_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
            self.lang_model = ModifiedOPTForCasualLM(model_config,
                                                     config, self.args) if 'opt' in config.pretrained_model_name_or_path \
                else ModifiedLlamaForCausalLM(model_config, config, self.args)
        else:
            self.lang_model = ModifiedOPTForCasualLM.from_pretrained(config.pretrained_model_name_or_path,
                                                                     config, self.args) if "opt" in config.pretrained_model_name_or_path \
                else ModifiedLlamaForCausalLM.from_pretrained(config.pretrained_model_name_or_path, config, self.args)

        self.lang_model.init_tokenizer(config.pretrained_model_name_or_path)

        self.hidden_size = self.lang_model.hidden_size
        self.model_type = self.lang_model.model_type

        # Panorama Encoding
        config.output_size = self.hidden_size
        self.img_embeddings = ImageEmbeddings(config, use_obj=args.enable_og, fuse_obj=args.fuse_obj)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.hidden_size)

        # global encoding
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, self.hidden_size)

        # local encoding
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        self.obj_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        if self.config.obj_feat_size > 0:
            self.og_head = nn.Sequential(
                nn.Linear(self.hidden_size, 100)
            ).to(self.lang_model.model_type)

            # Classfification from candidates
        self.out_head = nn.Sequential(
            nn.Linear(self.hidden_size, 100)
        ).to(self.lang_model.model_type)

        self.instruction = None
        self.history = None
        self.hist_vis = None

        self.drop_env = nn.Dropout(p=args.feat_dropout)

        logger.info("model type: {}".format(self.model_type))

        if args.enable_RL_A2C:
            #self.critic = Critic(args)
            self.critic = nn.Sequential(
                nn.Linear(4096, 768),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(768, 1),
            ).to(self.lang_model.model_type)
        if args.mlm:
            self.MLMhead = BertOnlyMLMHead(self.lang_model.config, layer_norm_eps=self.config.layer_norm_eps).to(self.lang_model.model_type)

    def rand_permute_cand_in_prompt(self, prompts, rand_perms):
        bs = len(prompts)
        for i in range(bs):
            rand_perm_list = rand_perms[i].numpy().tolist()

            tmp_prompt = prompts[i]
            start = tmp_prompt.find("### Candidate: ") + len("### Candidate: ")
            end = tmp_prompt.find("\n", start)
            filtered_prompt = tmp_prompt[start:end]
            ori_candidates = ["(" + item.strip() for item in filtered_prompt.split('(') if item != ''][
                             1:]  # remove stop option
            new_candidates = [ori_candidates[idx] for idx in rand_perm_list]
            new_ordered_candidates = ["(0) stop"] + [f"({j + 1})" + a[a.find(')') + 1:] for j, a in
                                                     enumerate(new_candidates)]
            prompts[i] = tmp_prompt[:start] + ' '.join(new_ordered_candidates) + tmp_prompt[end:]
        return prompts

    def rand_permute_cot_gt(self, target, cot_gt, rand_perms, unnavigable_size):
        bs = len(target)

        new_target = np.zeros(bs, dtype=np.int64)
        for i in range(bs):
            rand_perm_list = rand_perms[i].numpy().tolist()
            # print("target")
            # print(target[i])
            #print(rand_perm_list)
            if target[i] == 0:
                new_target[i] = 0
            elif target[i] == -100:
                new_target[i] = -100
            else:
                new_target[i] = rand_perm_list.index(target[i]-1-unnavigable_size[i]) + 1 # add stop dimension

            action_pred_replace_position = cot_gt[i].find('- Action Decision: ')+len('- Action Decision: ')
            if target[i] == -100:
                cot_gt[i] = cot_gt[i][:action_pred_replace_position] + "(" + str(0) + ")"
            else:
                cot_gt[i] = cot_gt[i][:action_pred_replace_position] + "(" + str(new_target[i]) + ")"
            # print(cot_gt[i])
        new_target = torch.from_numpy(new_target).cuda()
        return new_target, cot_gt

    def rand_permute_cot_gt_add_action(self, target, cot_gt, rand_perms, unnavigable_size):
        bs = len(target)
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy().tolist()

        new_target = np.zeros(bs, dtype=np.int64)
        for i in range(bs):
            rand_perm_list = rand_perms[i].numpy().tolist()
            print("target")
            print(target[i])
            #print(rand_perm_list)
            if target[i] == 0:
                new_target[i] = 0
            elif target[i] == -100:
                new_target[i] = -100
            else:
                new_target[i] = rand_perm_list.index(target[i]-1-unnavigable_size[i]) + 1 # add stop dimension

            #action_pred_replace_position = cot_gt[i].find('- Action Decision: ')+len('- Action Decision: ')
            if target[i] == -100:
                #cot_gt[i] = cot_gt[i][:action_pred_replace_position] + "(" + str(0) + ")"
                cot_gt[i] += f" Therefore the correct action decision is (0)."
            else:
                #cot_gt[i] = cot_gt[i][:action_pred_replace_position] + "(" + str(new_target[i]) + ")"
                cot_gt[i] += f" Therefore the correct action decision is ({new_target[i]})."
            # print(cot_gt[i])
        new_target = torch.from_numpy(new_target).cuda()
        return new_target, cot_gt

    def parse_output(self, output, fuse_embeds, cand_masks, inv_perms):
        bs = len(output)
        a_t = np.zeros(bs, dtype=np.int64)
        for i in range(bs):
            output[i] = output[i].strip()
            if "- Action Decision: (" in output[i] and ")" in output[i]:
                # print("output")
                # print(output[i])
                # print("inv_perms[i]")
                # print(inv_perms[i])
                # print("a_t[i]")
                # print(int(output[i][output[i].find("- Action Decision: (")+len("- Action Decision: (")]))
                inv_perm_list = inv_perms[i].numpy().tolist()
                if int(output[i][output[i].find("- Action Decision: (")+len("- Action Decision: (")]) < len(inv_perm_list):
                    a_t[i] = inv_perm_list[int(output[i][output[i].find("- Action Decision: (")+len("- Action Decision: (")])]
                    if a_t[i] != 0:
                        true_num = 0
                        for ind in range(cand_masks[i].size(0)):
                            if cand_masks[i][ind]:
                                true_num += 1
                                if true_num == a_t[i]:
                                    real_index = ind
                                    a_t[i] = real_index
                                    break
                else:
                    fuse_logits = torch.rand(fuse_embeds[i].shape[0]).cuda()
                    fuse_logits.masked_fill_(cand_masks[i].logical_not(), -float('inf'))
                    # print(fuse_logits)
                    nav_probs = torch.softmax(fuse_logits / self.args.temperature, 0)
                    c = torch.distributions.Categorical(nav_probs.float())
                    # a_t[i] = random.randint(0, len(nav_vpids[i])-1)
                    a_t[i] = c.sample().detach()
            else:
                # print("output")
                # print(output[i])
                # a_t[i] = random.randint(0, len(nav_vpids[i]) - 1)
                # print("nav_vpids[i])")
                # print(nav_vpids[i])
                # print("cand_masks[i])")
                # print(cand_masks[i])
                fuse_logits = torch.rand(fuse_embeds[i].shape[0]).cuda()
                fuse_logits.masked_fill_(cand_masks[i].logical_not(), -float('inf'))
                # print(fuse_logits)
                nav_probs = torch.softmax(fuse_logits / self.args.temperature, 0)
                c = torch.distributions.Categorical(nav_probs.float())
                # a_t[i] = random.randint(0, len(nav_vpids[i])-1)
                a_t[i] = c.sample().detach()

        return torch.from_numpy(a_t).cuda()

    def add_summariazation_in_prompt(self, prompts, cot_summarization, nav_vpids, rand_perms,land_token=None,dir_token=None,direction_of_gt=None,activate_fast=False,training=False):
        ## 0,1,2,3,4,5 => after random, actual:[0, 1, 4, 2, 5, 3]
        ## forward 3,5 => 3 is now at 5, 5 is now at 4 => forward 5,4

        bs = len(prompts)
        direction_landmark_dict = {}
        direction_mapping = {"Stop":0, "turn right": 1, "turn left": 2, "go forward": 3,
                             "go back": 4, "go up": 5, "go down": 6}

        direction_gt_mapping = {
            "in front of":'go forward',
            "behind":'go back',
            "to the right of":'turn right',
            "to the left of":'turn left',
            "above":'go up',
            "below":'go down'
        }

        # direction_gt_mapping = {
        #     'go forward':"in front of",
        #     'go back':"behind",
        #     'turn right':"to the right of",
        #     'turn left':"to the left of",
        #     'go up':"above",
        #     'go down':"below"
        # }

        dir_token_id_bs = []
        land_token_id_bs = []
        land_predict_label = np.zeros(bs, dtype=np.int64)
        dir_predict_label = np.zeros(bs, dtype=np.int64)
        land_token_mask = torch.ones((bs, len(list(direction_mapping.keys())))).bool().cuda()
        dir_token_mask = torch.ones((bs, len(list(direction_mapping.keys())))).bool().cuda()

        for i in range(bs):
            tmp_prompt = prompts[i]
            start = tmp_prompt.find("### Candidate: ") + len("### Candidate: ")
            end = tmp_prompt.find("\n", start)
            #print(rand_perms)
            rand_perm_list = [0] + [item + 1 for item in rand_perms[i].cpu().numpy().tolist()]  ### put stop in rand_perm_list, because cot_summarization[i][direction]['cand_index'] records cand index containing stop
            # ori_oder_list = list(range(nav_vpids[1:]))
            directions = list(cot_summarization[i].keys())
            #directions = list (direction_gt_mapping.keys())
            direction_texts = []
            dir_token_id = []
            land_token_id = []

            for direction in directions:
                direction_landmark_dict[direction] = f"[{', '.join(cot_summarization[i][direction]['landmarks'])}]"
                #direction_landmark_dict[direction] = f"[{', '.join(cot_summarization[i][direction_gt_mapping[direction]]['landmarks'])}]"

                if self.args.cot_v4_only_direction:
                    direction_texts.append(
                        f"{direction.capitalize()} are Candidates {' '.join([f'({rand_perm_list.index(x)})' for x in cot_summarization[i][direction]['cand_index']])}.")
                else:
                    direction_texts.append(
                        f"{direction.capitalize()} are Candidates {' '.join([f'({rand_perm_list.index(x)})' for x in cot_summarization[i][direction]['cand_index']])} [{', '.join(cot_summarization[i][direction]['landmarks'])}].")
            summary = f"\n### Summarization: {' '.join(direction_texts)}"

            # print("activate_fast old")
            # print(activate_fast)

            if self.args.remove_summarization:
                # print('branch 1')
                activate_fast = True
            # if self.args.random_slow_and_fast_prob and not training:
            #     print('self.args.random_slow_and_fast_prob')
            #     print(self.args.random_slow_and_fast_prob)
            #     print('training')
            #     print(training)
            #     print('not training')
            #     print(not training)


            # print("activate_fast")
            # print(activate_fast)
            if activate_fast:
                prompts[i] = tmp_prompt[:end] + tmp_prompt[end:]
            else:
                prompts[i] = tmp_prompt[:end] + summary + tmp_prompt[end:]

        return prompts, direction_landmark_dict

            # Turn right are Candidate (1) containing [landmark 1, landmark 2, ...]. Turn left are Candidate (2) containing [landmark 1, landmark 2, ...]. Go forward are... Go back are... Go up are... Go down are...

    def forward(self, mode: str, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'panorama':  # batch['view_img_fts'] [B, 36, D=768] --> dropout
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            return self.img_embeddings.forward_panorama_per_step(
                batch['view_img_fts'],
                batch['view_lens'],
                batch['loc_fts'],
                batch['nav_types'],
                batch['obj_img_fts'],
                batch['obj_lens'],
                batch['obj_loc_fts'],
            )

        elif mode == 'navigation':
            return self.forward_navigation(mode, batch, **kwargs)

        elif mode == "summarization" or mode == 'embodied_qa':
            return self.forward_summarization(mode, batch, **kwargs)

        elif mode == "3dqa":
            return self.forward_3dqa(mode, batch, **kwargs)

        elif mode == 'object_grounding':
            return self.forward_object_grounding(mode, batch, **kwargs)

        elif mode == 'spatial_relation':
            return self.forward_spatial_relation(mode, batch, **kwargs)

        elif mode == 'navigation_cot':
            return self.forward_navigation_cot(mode, batch, **kwargs)

        elif mode == 'navigation_once_forward_cot_navigation':
            return self.forward_navigation_once_forward_cot_navigation(mode, batch, **kwargs)

        elif mode == 'critic':
            return self.forward_critic(mode, **kwargs)
        else:
            raise NotImplementedError('wrong mode: %s' % mode)

    def forward_critic(self,  mode,
            **kwargs):
        state = kwargs['state']
        output = self.critic(state)
        #print(f"before: output size {output.size()}")
        output = output.squeeze(1)
        #print(f"after: output size {output.size()}")
        return output

    def forward_navigation(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
            gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
            = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
            batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

        # global branch [B, Nums, D=768]
        gmap_embeds = torch.zeros_like(gmap_img_embeds)
        for b_ix in range(len(data_type)):
            gmap_embeds[b_ix:b_ix + 1] = gmap_img_embeds[b_ix:b_ix + 1] + \
                                         self.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix + 1]) + \
                                         self.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix + 1])

        ##### local branch #####
        vp_img_embeds, vp_pos_fts, vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_nav_masks'], batch['vp_cand_vpids']

        pano_masks = batch['pano_masks']

        vp_embeds = torch.zeros_like(vp_img_embeds)
        for b_ix in range(len(data_type)):
            vp_embeds[b_ix:b_ix + 1] = vp_img_embeds[b_ix:b_ix + 1] \
                                       + self.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix + 1])

        ##### fuse embeds #####
        gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)
        cand_token_type_ids = torch.zeros((gmap_embeds.shape[0], gmap_embeds.shape[1])).int().to(gmap_embeds.device)

        local_vp_embeds = vp_embeds
        local_vp_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)

        fuse_embeds = torch.clone(gmap_embeds)

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_vp_embeds[i, j]
                    else:
                        tmp[cand_vpid] = local_vp_embeds[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fuse_embeds[i, j] += tmp[vp]
                    else:
                        # fuse_embeds[i, j] += bw_logits
                        cand_token_type_ids[i, j] = 1

        fuse_embeds += self.token_type_embeddings(cand_token_type_ids).to(fuse_embeds.device)
        fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

        cand_masks = torch.clone(gmap_masks & gmap_visited_masks.logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        # print(f"text_input['input_ids']:{text_input['input_ids'].size()}")

        # cand_embeds = fuse_embeds[cand_masks]  # .to(self.model_type)
        cand_embeds = []
        inv_perms = []
        rand_perms = []
        for bn in range(batch_size):
            # random permute
            cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]
            rand_perm = torch.randperm(cand_embed.shape[0])
            rand_perms.append(rand_perm)
            inv_perm = torch.arange(cand_embed.shape[0])
            inv_perm[rand_perm] = torch.arange(
                cand_embed.shape[0])  # put inv_perm[idx] of idx in rand_perm to rand_perm[idx]'s position
            inv_perms.append(inv_perm)
            cand_embeds.append(cand_embed[rand_perm])  # remove stop features
        cand_embeds = torch.cat(cand_embeds, dim=0)

        if self.args.add_cand_landmark:
            batch["prompts"] = self.rand_permute_cand_in_prompt(batch["prompts"], rand_perms)

        # print(f"{batch['prompts']}")

        text_input = self.lang_model.tokenize(batch["prompts"]).to(fuse_embeds.device)


        output = self.lang_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            cand_vis=cand_embeds,
            hist_vis=hist_vis_input,
        )
        loss, hidden_states = output.loss, output.hidden_states

        fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
            fuse_embeds.device).to(self.model_type)

        predictions = self.out_head(hidden_states[text_input['input_ids'] == self.lang_model.cls_token_id[0]])

        for i in range(batch_size):
            fuse_logits[i][cand_masks[i]] = torch.cat(
                [predictions[i, 0:1], predictions[i, 1:cand_nums[i]][inv_perms[i]]], dim=0)

        fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))

        return {
            'fuse_embeds': fuse_embeds.detach(),
            'fuse_logits': fuse_logits,
        }

    def forward_summarization(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            **kwargs
    ) -> Dict[str, Any]:

        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        vp_img_embeds, vp_pos_fts, \
            vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], \
                batch['vp_nav_masks'], batch['vp_cand_vpids']

        # remove `stop`
        vp_img_embeds = vp_img_embeds[:, 1:, :]
        vp_nav_masks = vp_nav_masks[:, 1:]

        vp_pos_fts = torch.zeros(vp_img_embeds.shape[:2] + (14,), dtype=torch.float).to(vp_img_embeds.device)
        token_type_ids = torch.zeros(vp_img_embeds.shape[:2], dtype=torch.int).to(vp_img_embeds.device)
        vp_img_embeds += self.vp_pos_embeddings(vp_pos_fts)
        vp_img_embeds += self.token_type_embeddings(token_type_ids)

        instruction = batch['instruction']
        labels = batch['answer']
        history = batch['history']
        hist_vis = batch['hist_vis']
        data_type = batch['data_type']
        hist_vis_input = []

        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]
        cand_nums = vp_nav_masks.sum(1)

        all_text = []

        for bn in range(batch_size):
            prompt = batch["prompts"][bn]
            if data_type[0] == 'eqa' or data_type[0] == 'fgr2r':
                label = labels[bn] + f"{self.lang_model.tokenizer.eos_token}"
            else:
                label = batch["instruction"][bn] + f"{self.lang_model.tokenizer.eos_token}"
            if training:
                all_text.append([prompt, label])
            else:
                all_text.append(prompt)

        text_input = self.lang_model.tokenize(all_text).to(vp_img_embeds.device)
        if training:
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=vp_img_embeds[vp_nav_masks],
                hist_vis=hist_vis_input,
            )
            loss, logits, hidden_states = outputs.loss, outputs.logits, outputs.hidden_states
            outputs = {
                "loss": loss
            }
        else:
            trie = kwargs.get('trie', None)
            logits_processor = [TrieLogitsProcessor(trie)] if trie is not None else []

            generate_ids = self.lang_model.generate(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=vp_img_embeds[vp_nav_masks],
                hist_vis=hist_vis_input,
                bos_token_id=self.lang_model.tokenizer.bos_token_id,
                eos_token_id=self.lang_model.tokenizer.eos_token_id,
                pad_token_id=self.lang_model.tokenizer.unk_token_id,
                max_new_tokens=50,
                do_sample=False,
                logits_processor=logits_processor
            ).tolist()

            generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(generate_ids)]
            generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
            outputs = {
                "generated_sentences": generated_sentences
            }

        return outputs

    def forward_3dqa(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        batch_size = len(batch['question'])
        data_type = batch['data_type']
        all_text = []
        for bn in range(batch_size):
            prompt = batch["prompts"][bn]
            if training:
                ans = batch["answers"][bn][0] + f"{self.lang_model.tokenizer.eos_token}"
                all_text.append([prompt, ans])
            else:
                all_text.append(prompt)

        view_img_fts = pad_tensors_wgrad([batch["features"][bn] for bn in range(batch_size)])
        view_lens = torch.tensor([batch["features"][bn].shape[0] for bn in range(batch_size)]).to(view_img_fts.device)
        pano_outputs = self.img_embeddings.forward_panorama_per_step(
            view_img_fts=view_img_fts,
            view_lens=view_lens,
        )
        pano_embeds, pano_masks = pano_outputs["pano_embeds"], pano_outputs["pano_masks"]
        vp_pos_fts = torch.zeros(pano_embeds.shape[:2] + (14,), dtype=torch.float).to(pano_embeds.device)
        token_type_ids = torch.zeros(pano_embeds.shape[:2], dtype=torch.int).to(pano_embeds.device)
        pano_embeds += self.vp_pos_embeddings(vp_pos_fts)
        pano_embeds += self.token_type_embeddings(token_type_ids)

        text_input = self.lang_model.tokenize(all_text).to(pano_embeds.device)
        if training:
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=pano_embeds[pano_masks],
            )
        else:

            generate_ids = self.lang_model.generate(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=pano_embeds[pano_masks],
                bos_token_id=self.lang_model.tokenizer.bos_token_id,
                eos_token_id=self.lang_model.tokenizer.eos_token_id,
                pad_token_id=self.lang_model.tokenizer.unk_token_id,
                **kwargs
            ).tolist()

            generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(generate_ids)]
            generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
            outputs = {
                "generated_sentences": generated_sentences
            }

        return outputs

    def forward_object_grounding(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        obj_embeds, obj_masks, obj_loc_fts = batch['obj_embeds'], batch['obj_masks'], batch['obj_loc_fts']

        batch_size = obj_embeds.size(0)
        obj_embeds = obj_embeds + self.obj_pos_embeddings(obj_loc_fts)

        cand_nums = obj_masks.sum(dim=1) + 1  # add not exist

        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        text_input = self.lang_model.tokenize(batch["prompts"]).to(obj_embeds.device)
        output = self.lang_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            cand_vis=obj_embeds[obj_masks],
            hist_vis=hist_vis_input,
        )
        loss, hidden_states = output.loss, output.hidden_states

        predictions = self.out_head(hidden_states[text_input['input_ids'] == self.lang_model.cls_token_id[0]])
        for i in range(batch_size):
            predictions[i, cand_nums[i]:] = float('-inf')

        return {
            'obj_logits': predictions
        }

    ### newly added
    def forward_spatial_relation(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            **kwargs
    ) -> Dict[str, Any]:

        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        vp_img_embeds, vp_pos_fts, \
            vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], \
                batch['vp_nav_masks'], batch['vp_cand_vpids']

        # remove `stop`
        vp_img_embeds = vp_img_embeds[:, 1:, :]
        vp_nav_masks = vp_nav_masks[:, 1:]

        vp_pos_fts = torch.zeros(vp_img_embeds.shape[:2] + (14,), dtype=torch.float).to(vp_img_embeds.device)
        token_type_ids = torch.zeros(vp_img_embeds.shape[:2], dtype=torch.int).to(vp_img_embeds.device)
        vp_img_embeds += self.vp_pos_embeddings(vp_pos_fts)
        vp_img_embeds += self.token_type_embeddings(token_type_ids)

        instruction = batch['instruction']
        labels = batch['QA_cand_GTs']
        data_type = batch['data_type']
        hist_vis_input = []

        cand_nums = vp_nav_masks.sum(1)

        all_text = []

        for bn in range(batch_size):
            prompt = batch["prompts"][bn]
            label = labels[bn]
            # if data_type[0] == 'eqa' or data_type[0] == 'fgr2r':
            #     label = labels[bn] + f"{self.lang_model.tokenizer.eos_token}"
            # else:
            #     label = batch["instruction"][bn] + f"{self.lang_model.tokenizer.eos_token}"
            if training:
                all_text.append([prompt, label])
            else:
                all_text.append(prompt)

        text_input = self.lang_model.tokenize(all_text).to(vp_img_embeds.device)
        if training:
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=vp_img_embeds[vp_nav_masks],
                hist_vis=hist_vis_input,
            )
            loss, logits, hidden_states = outputs.loss, outputs.logits, outputs.hidden_states
            outputs = {
                "loss": loss
            }
        else:
            trie = kwargs.get('trie', None)
            logits_processor = [TrieLogitsProcessor(trie)] if trie is not None else []

            generate_ids = self.lang_model.generate(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=vp_img_embeds[vp_nav_masks],
                hist_vis=hist_vis_input,
                bos_token_id=self.lang_model.tokenizer.bos_token_id,
                eos_token_id=self.lang_model.tokenizer.eos_token_id,
                pad_token_id=self.lang_model.tokenizer.unk_token_id,
                max_new_tokens=50,
                do_sample=False,
                logits_processor=logits_processor
            ).tolist()

            generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(generate_ids)]
            generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
            outputs = {
                "generated_sentences_spatial_relation": generated_sentences
            }

        return outputs

    def forward_navigation_cot(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            rand_perms=None,
            **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
            gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
            = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
            batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

        # global branch [B, Nums, D=768]
        gmap_embeds = torch.zeros_like(gmap_img_embeds)
        for b_ix in range(len(data_type)):
            gmap_embeds[b_ix:b_ix + 1] = gmap_img_embeds[b_ix:b_ix + 1] + \
                                         self.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix + 1]) + \
                                         self.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix + 1])

        ##### local branch #####
        vp_img_embeds, vp_pos_fts, vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_nav_masks'], batch['vp_cand_vpids']

        pano_masks = batch['pano_masks']

        vp_embeds = torch.zeros_like(vp_img_embeds)
        for b_ix in range(len(data_type)):
            vp_embeds[b_ix:b_ix + 1] = vp_img_embeds[b_ix:b_ix + 1] \
                                       + self.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix + 1])

        ##### fuse embeds #####
        gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)
        cand_token_type_ids = torch.zeros((gmap_embeds.shape[0], gmap_embeds.shape[1])).int().to(gmap_embeds.device)

        local_vp_embeds = vp_embeds
        local_vp_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)

        fuse_embeds = torch.clone(gmap_embeds)

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_vp_embeds[i, j]
                    else:
                        tmp[cand_vpid] = local_vp_embeds[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fuse_embeds[i, j] += tmp[vp]
                    else:
                        # fuse_embeds[i, j] += bw_logits
                        cand_token_type_ids[i, j] = 1

        fuse_embeds += self.token_type_embeddings(cand_token_type_ids).to(fuse_embeds.device)
        fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

        cand_masks = torch.clone(gmap_masks & gmap_visited_masks.logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        text_input = self.lang_model.tokenize(batch["prompts"]).to(fuse_embeds.device)

        # cand_embeds = fuse_embeds[cand_masks]  # .to(self.model_type)
        cand_embeds = []
        inv_perms = []
        if rand_perms is None:
            rand_perms = []
            for bn in range(batch_size):
                # random permute
                cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]
                rand_perm = torch.randperm(cand_embed.shape[0])
                rand_perms.append(rand_perm)
                inv_perm = torch.arange(cand_embed.shape[0])
                inv_perm[rand_perm] = torch.arange(cand_embed.shape[0])
                inv_perms.append(inv_perm)
                cand_embeds.append(cand_embed[rand_perm])  # remove stop features
            cand_embeds = torch.cat(cand_embeds, dim=0)
        else:
            for bn in range(batch_size):
                # random permute
                cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]
                rand_perm = rand_perms[bn]
                #rand_perms.append(rand_perm)
                inv_perm = torch.arange(cand_embed.shape[0])
                inv_perm[rand_perm] = torch.arange(cand_embed.shape[0])
                inv_perms.append(inv_perm)
                cand_embeds.append(cand_embed[rand_perm])  # remove stop features
            cand_embeds = torch.cat(cand_embeds, dim=0)

        all_text = []

        if self.args.add_cand_landmark:
            batch["prompts"] = self.rand_permute_cand_in_prompt(batch["prompts"], rand_perms)

        if self.args.cot_summarization:
            batch["prompts"], direction_landmark_dict = self.add_summariazation_in_prompt(batch["prompts"], batch["cot_summarization"],
                                                                 batch['gmap_vpids'], rand_perms)
        else:
            direction_landmark_dict = None

        for bn in range(batch_size):
            prompt = batch["prompts"][bn]

            if training:
                label = batch['navigation_cot_gt'][bn] + f"{self.lang_model.tokenizer.eos_token}"
                all_text.append([prompt, label])
                if self.args.check_cot_input_gt:
                    print(f"\n{prompt}{label}")
            else:
                all_text.append(prompt)

        text_input = self.lang_model.tokenize(all_text).to(vp_img_embeds.device)
        if training:
            labels = text_input['input_ids'].clone()
            labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=cand_embeds,
                hist_vis=hist_vis_input,
            )
            loss, logits, hidden_states = outputs.loss, outputs.logits, outputs.hidden_states
            outputs = {
                'fuse_embeds': fuse_embeds.detach(),
                "loss": loss,
                "prompts": batch["prompts"]
            }
        else:
            trie = kwargs.get('trie', None)
            logits_processor = [TrieLogitsProcessor(trie)] if trie is not None else []

            generate_ids = self.lang_model.generate(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                cand_vis=cand_embeds,
                hist_vis=hist_vis_input,
                bos_token_id=self.lang_model.tokenizer.bos_token_id,
                eos_token_id=self.lang_model.tokenizer.eos_token_id,
                pad_token_id=self.lang_model.tokenizer.unk_token_id,
                max_new_tokens=100,
                do_sample=False,
                logits_processor=logits_processor
            ).tolist()

            generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(generate_ids)]
            generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
            outputs = {
                "generated_sentences_navigation_cot": generated_sentences,
                "rand_perms": rand_perms,
                "direction_landmark_dict": direction_landmark_dict,
                "prompts": batch["prompts"]
            }

        return outputs

    def forward_navigation_once_forward_cot_navigation(
            self,
            mode,
            batch: Dict[str, Any],
            training: bool = True,
            **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
            gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
            = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
            batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

        # global branch [B, Nums, D=768]
        gmap_embeds = torch.zeros_like(gmap_img_embeds)
        for b_ix in range(len(data_type)):
            gmap_embeds[b_ix:b_ix + 1] = gmap_img_embeds[b_ix:b_ix + 1] + \
                                         self.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix + 1]) + \
                                         self.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix + 1])

        ##### local branch #####
        vp_img_embeds, vp_pos_fts, vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_nav_masks'], batch['vp_cand_vpids']

        pano_masks = batch['pano_masks']

        vp_embeds = torch.zeros_like(vp_img_embeds)
        for b_ix in range(len(data_type)):
            vp_embeds[b_ix:b_ix + 1] = vp_img_embeds[b_ix:b_ix + 1] \
                                       + self.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix + 1])

        ##### fuse embeds #####
        gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)
        cand_token_type_ids = torch.zeros((gmap_embeds.shape[0], gmap_embeds.shape[1])).int().to(gmap_embeds.device)

        local_vp_embeds = vp_embeds
        local_vp_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)

        fuse_embeds = torch.clone(gmap_embeds)

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_vp_embeds[i, j]
                    else:
                        tmp[cand_vpid] = local_vp_embeds[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fuse_embeds[i, j] += tmp[vp]
                    else:
                        # fuse_embeds[i, j] += bw_logits
                        cand_token_type_ids[i, j] = 1

        fuse_embeds += self.token_type_embeddings(cand_token_type_ids).to(fuse_embeds.device)
        fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

        cand_masks = torch.clone(gmap_masks & gmap_visited_masks.logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        # text_input = self.lang_model.tokenize(batch["prompts"]).to(fuse_embeds.device)

        # cand_embeds = fuse_embeds[cand_masks]  # .to(self.model_type)
        cand_embeds = []
        inv_perms = []
        rand_perms = []
        unnavigable_size = []
        for bn in range(batch_size):
            # random permute
            cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]

            if self.args.visualize:
                rand_perm = torch.arange(cand_embed.shape[0])
            else:
                rand_perm = torch.randperm(cand_embed.shape[0])
            rand_perms.append(rand_perm)
            inv_perm = torch.arange(cand_embed.shape[0])
            inv_perm[rand_perm] = torch.arange(cand_embed.shape[0])
            inv_perms.append(inv_perm)
            cand_embeds.append(cand_embed[rand_perm])  # remove stop features
        cand_embeds = torch.cat(cand_embeds, dim=0)

        all_text = []

        if self.args.add_cand_landmark:
            batch["prompts"] = self.rand_permute_cand_in_prompt(batch["prompts"], rand_perms)

        if self.args.cot_summarization:

            batch["prompts"], direction_landmark_dict = self.add_summariazation_in_prompt(batch["prompts"], batch["cot_summarization"],
                                                                 batch['gmap_vpids'], rand_perms,training=training)
            print("new prompts")
            print(batch["prompts"])

        if self.args.mlm and training:
            label_text_land = []
            label_text_dir = []

        for bn in range(batch_size):
            prompt = batch["prompts"][bn]

            # if training:
            #     label = batch['navigation_cot_gt'][bn] + f"{self.lang_model.tokenizer.eos_token}"
            #     if self.args.check_cot_input_gt:
            #         print(f"\n{prompt}{label}")
            #     all_text.append([prompt, label])
            # else:
            #     all_text.append(prompt)
            if training:
                if self.args.mlm:
                    #label = f"{self.lang_model.tokenizer.eos_token}"
                    prompt += f"{self.lang_model.tokenizer.eos_token}"
                    all_text.append(prompt)
                    print('navigation_cot_gt land')
                    print(batch['navigation_cot_gt'])
                    #label_text_land_bs = " ".join(batch['navigation_cot_gt'][bn]["land"])
                    #label_text_land_bs = label_text_land_bs[:-1]
                    label_text_land_bs = batch['navigation_cot_gt'][bn]["land"]
                    print('label_text_land_bs')
                    print(label_text_land_bs)
                    label_text_land.append(label_text_land_bs)
                    label_text_dir_bs = batch['navigation_cot_gt'][bn]["dir_1"] + " " + batch['navigation_cot_gt'][bn]["dir_2"]
                    label_text_dir.append(label_text_dir_bs)
                else:
                    label = batch['navigation_cot_gt'][bn] + f"{self.lang_model.tokenizer.eos_token}"
                    # if self.args.check_cot_input_gt:
                    #     print(f"\n{prompt}{label}")
                    all_text.append([prompt, label])
            else:
                all_text.append(prompt)

        text_input = self.lang_model.tokenize(all_text).to(vp_img_embeds.device)
        if self.args.mlm and training:
            label_text_input_land = self.lang_model.tokenize(label_text_land).to(vp_img_embeds.device)
            label_text_input_dir = self.lang_model.tokenize(label_text_dir).to(vp_img_embeds.device)
            label_text_input_land_new = label_text_input_land['input_ids']
            label_text_input_dir_new = label_text_input_dir['input_ids']



        multiple_sample_cot = kwargs.get('multiple_sample_cot', False)
        get_cot_output = kwargs.get('get_cot_output', False)

        if training:
            # if self.args.cot_output_as_supervision:
            #     labels = None
            # else:
            if self.args.mlm:
                labels = text_input['input_ids'].clone()
                labels[:, :] = -100
                print('landmark_token_id')
                print(self.lang_model.landmark_token_id)
                print('direction_token_id')
                print(self.lang_model.direction_token_id)
                for bs in range(len(text_input['input_ids'])):
                    print('text_input')
                    #print(text_input['input_ids'][bs])
                    # first token is begin token
                    print(label_text_input_land_new[bs])
                    print(label_text_input_dir_new[bs])

                    for i , token in enumerate(text_input['input_ids'][bs]):
                        # if token in self.lang_model.landmark_token_id:
                        #     if token == self.lang_model.landmark_token_id[0]:
                        #         labels[bs][i] = label_text_input_land['input_ids'][bs][1]
                        #     elif token == self.lang_model.landmark_token_id[1]:
                        #         labels[bs][i] = label_text_input_land['input_ids'][bs][2]
                        #     elif token == self.lang_model.landmark_token_id[2]:
                        #         labels[bs][i] = label_text_input_land['input_ids'][bs][3]
                        #     elif token == self.lang_model.landmark_token_id[3]:
                        #         labels[bs][i] = label_text_input_land['input_ids'][bs][4]
                        #     elif token == self.lang_model.landmark_token_id[4]:
                        #         labels[bs][i] = label_text_input_land['input_ids'][bs][5]
                        if token in self.lang_model.direction_token_id:
                            if token == self.lang_model.direction_token_id[0]:
                                labels[bs][i] = label_text_input_dir_new[bs][1]
                            elif token == self.lang_model.direction_token_id[1]:
                                labels[bs][i] = label_text_input_dir_new[bs][2]
                        elif token == self.lang_model.landmark_token_id[0]:
                            if len(label_text_input_land_new[bs][1:]) < self.args.land_token_region_length:
                                labels[bs][i:i + len(label_text_input_land_new[bs][1:])] = label_text_input_land_new[bs][1:]
                                labels[bs][i+len(label_text_input_land_new[bs][1:]):i+self.args.land_token_region_length] = self.lang_model.landmark_token_id[1]
                            else:
                                labels[bs][i:i + self.args.land_token_region_length] = label_text_input_land_new[bs][1:self.args.land_token_region_length+1]
                            break
                print('labels')
                print(labels)
            else:
                labels = text_input['input_ids'].clone()
                labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100

            outputs = self.lang_model(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                labels=labels,
                cand_vis=cand_embeds,
                hist_vis=hist_vis_input,
            )
            loss, logits, hidden_states = outputs.loss, outputs.logits, outputs.hidden_states

            if self.args.mlm:
                #print('self.lang_model.config.vocab_size')
                #print(self.lang_model.config.vocab_size)
                #print(self.config.vocab_size)
                masked_output = self._compute_masked_hidden(hidden_states, labels!=-100)
                mlm_prediction_scores = self.MLMhead(masked_output)
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = mlm_prediction_scores.view(-1, self.lang_model.config.vocab_size)
                shift_labels = labels[labels!=-100].view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                #loss = F.cross_entropy(mlm_prediction_scores, labels[labels!=-100])

            # test = torch.arange(logits.size(-1)).unsqueeze(0).unsqueeze(0).repeat(1, logits.size(1), 1).to(
            #     logits.device)
            # x = test[logits == -torch.inf]

            fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                fuse_embeds.device).to(self.model_type)
            if self.args.action_first_in_gt:
                # res = outputs['sequences'][:, :-1] == self.lang_model.cls_token_id[0]
                # res_check = torch.tensor([False] * batch_size, dtype=torch.bool)
                # hidden_states = torch.cat(outputs['hidden_states'], dim=1)
                # for i in range(batch_size):
                #     res_check[i] = res[i].any()
                # if res_check.all():
                #     predictions = self.out_head(
                #         hidden_states[res])
                # else:
                #     predictions = self.out_head(outputs['hidden_states'][-1].squeeze(1))
                predictions = self.out_head(hidden_states[text_input['input_ids'].size(1)+1])
            else:
                # predictions = self.out_head(hidden_states[-2])
                predictions = self.out_head(hidden_states[text_input['input_ids'] == self.lang_model.cls_token_id[0]])

            # outputs = {
            #     'fuse_embeds': fuse_embeds.detach(),
            #     "loss": loss
            # }
            #
            # loss, hidden_states = output.loss, output.hidden_states
            for i in range(batch_size):
                fuse_logits[i][cand_masks[i]] = torch.cat(
                    [predictions[i, 0:1], predictions[i, 1:cand_nums[i]][inv_perms[i]]], dim=0)

            fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))

            land_predict_logit = None
            dir_predict_logit = None
            land_predict_label = None
            dir_predict_label = None
            return {
                'fuse_embeds': fuse_embeds.detach(),
                'fuse_logits': fuse_logits,
                'logits': outputs.logits,
                "loss": loss,
                "prompts": batch["prompts"],
                "land_predict_logit": land_predict_logit,
                "dir_predict_logit":dir_predict_logit,
                "land_predict_gt":land_predict_label,
                "dir_predict_gt":dir_predict_label,
                "direction_landmark_dict": direction_landmark_dict if self.args.cot_summarization else None,
                "rand_perms": rand_perms if self.args.cot_summarization else None,
                "cls_hidden_state": hidden_states[text_input['input_ids'] == self.lang_model.cls_token_id[0]] if self.args.enable_RL_A2C else None,
            }

        else:
            if self.args.cot_v4:
                should_generate_cot = (
                    self.args.visualize
                    or getattr(self.args, "visualize_cot", False)
                    or getattr(self.args, "enable_action_reasoning_f1", False)
                )
                if should_generate_cot:
                    trie = kwargs.get('trie', None)
                    logits_processor = [TrieLogitsProcessor(trie)] if trie is not None else []

                    if multiple_sample_cot:
                        num_return_sequences = self.args.cot_sample_return_sequences
                        do_sample = True
                        temperature = self.args.cot_sample_temperature
                    else:
                        num_return_sequences = 1
                        do_sample = False
                        temperature = 0

                    output = self.lang_model.generate(
                        input_ids=text_input['input_ids'],
                        attention_mask=text_input['attention_mask'],
                        cand_vis=cand_embeds,
                        hist_vis=hist_vis_input,
                        bos_token_id=self.lang_model.tokenizer.bos_token_id,
                        eos_token_id=self.lang_model.tokenizer.eos_token_id,
                        pad_token_id=self.lang_model.tokenizer.unk_token_id,
                        max_new_tokens=500,
                        do_sample=do_sample,
                        temperature = temperature,
                        logits_processor=logits_processor,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        num_return_sequences=num_return_sequences
                    )
                    # ).tolist()

                    generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(output['sequences'])]
                    # generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                    # clean_up_tokenization_spaces=False)
                    generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids,
                                                                                 skip_special_tokens=False,
                                                                                 clean_up_tokenization_spaces=False)

                    action = None

                    if multiple_sample_cot:
                        # fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                        #     fuse_embeds.device).to(self.model_type)
                        fuse_logits=torch.zeros((num_return_sequences, fuse_embeds.shape[1])).to(
                            fuse_embeds.device).to(self.model_type)
                    else:
                        fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                            fuse_embeds.device).to(self.model_type)

                    # res = text_input['input_ids'] == self.lang_model.cls_token_id[0]
                    # res_check = torch.tensor([False] * batch_size, dtype=torch.bool)
                    # for i in range(batch_size):
                    #     res_check[i] = res[i].any()
                    # if res_check.all():
                    #     predictions = self.out_head(
                    #         output['hidden_states'][text_input['input_ids'] == self.lang_model.cls_token_id[0]])
                    # else:
                    #     predictions = self.out_head(output['hidden_states'][-2].squeeze(1))
                    #



                    if multiple_sample_cot:
                        cls_tok_idx = text_input['input_ids'] == self.lang_model.cls_token_id[0]
                        hidden_states = output['hidden_states'][0]
                        predictions = self.out_head(hidden_states[cls_tok_idx.repeat(5,1)])
                        cand_masks = cand_masks.repeat(5, 1) ##
                        cand_nums = cand_masks.sum(dim=-1)
                        inv_perms = [inv_perms[0] for i in range(5)]
                        # for i in range(fuse_embeds.shape[0]):
                        for i in range(num_return_sequences):
                            fuse_logits[i][cand_masks[i]] = torch.cat(
                                [predictions[i, 0:1], predictions[i, 1:cand_nums[i]][inv_perms[i]]], dim=0)

                        fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))
                        print(f"fuse_logits: {fuse_logits}")
                    else:
                        res = output['sequences'][:, :-1] == self.lang_model.cls_token_id[0]
                        res_check = torch.tensor([False] * batch_size, dtype=torch.bool)
                        hidden_states = torch.cat(output['hidden_states'], dim=1)
                        for i in range(batch_size):
                            res_check[i] = res[i].any()
                        if res_check.all():
                            predictions = self.out_head(
                                hidden_states[res])
                            print("Correct predictions")
                        else:
                            predictions = self.out_head(output['hidden_states'][-1].squeeze(1))

                else:
                    assert multiple_sample_cot == False, "multiple_sample_cot is unnecessary False when cot_v4!! cotv4 places <CLS> before cot, doing sample here won't change the actions"

                    generated_sentences = None
                    action = None
                    output = self.lang_model(
                        input_ids=text_input['input_ids'],
                        attention_mask=text_input['attention_mask'],
                        cand_vis=cand_embeds,
                        hist_vis=hist_vis_input,
                    )
                    loss, hidden_states = output.loss, output.hidden_states

                    fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                        fuse_embeds.device).to(self.model_type)

                    predictions = self.out_head(hidden_states[text_input['input_ids'] == self.lang_model.cls_token_id[0]])
            else:
                trie = kwargs.get('trie', None)
                logits_processor = [TrieLogitsProcessor(trie)] if trie is not None else []

                if multiple_sample_cot:
                    num_return_sequences = self.args.cot_sample_return_sequences
                    do_sample = True
                    temperature = self.args.cot_sample_temperature
                else:
                    num_return_sequences = 1
                    do_sample = False
                    temperature = 0

                output = self.lang_model.generate(
                    input_ids=text_input['input_ids'],
                    attention_mask=text_input['attention_mask'],
                    cand_vis=cand_embeds,
                    hist_vis=hist_vis_input,
                    bos_token_id=self.lang_model.tokenizer.bos_token_id,
                    eos_token_id=self.lang_model.tokenizer.eos_token_id,
                    pad_token_id=self.lang_model.tokenizer.unk_token_id,
                    max_new_tokens=500,
                    do_sample=do_sample,
                    temperature=temperature,
                    logits_processor=logits_processor,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    num_return_sequences=num_return_sequences
                )
                # ).tolist()

                generate_ids = [s[text_input["input_ids"].shape[1]:] for i, s in enumerate(output['sequences'])]
                # generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                # clean_up_tokenization_spaces=False)
                generated_sentences = self.lang_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=False,
                                                                             clean_up_tokenization_spaces=False)

                action = None
                if multiple_sample_cot:
                    fuse_logits = torch.zeros((num_return_sequences, fuse_embeds.shape[1])).to(
                        fuse_embeds.device).to(self.model_type)
                else:
                    fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                        fuse_embeds.device).to(self.model_type)


                if multiple_sample_cot:

                    res = output['sequences'][:, :-1] == self.lang_model.cls_token_id[0]  # last token is eos
                    res_check = torch.tensor([False] * res.size(0), dtype=torch.bool)
                    hidden_states = torch.cat(output['hidden_states'], dim=1)
                    for i in range(res.size(0)):
                        res_check[i] = res[i].any()
                    if res_check.all():
                        predictions = self.out_head(
                            hidden_states[res])
                    else:
                        if self.args.action_first_in_gt:
                            predictions = self.out_head(output['hidden_states'][1].squeeze(1))
                        else:
                            predictions = self.out_head(output['hidden_states'][-1].squeeze(1))
                    cand_masks = cand_masks.repeat(5, 1)
                    cand_nums = cand_masks.sum(dim=-1)
                    inv_perms = [inv_perms[0] for i in range(5)]
                    for i in range(num_return_sequences):
                        fuse_logits[i][cand_masks[i]] = torch.cat(
                            [predictions[i, 0:1], predictions[i, 1:cand_nums[i]][inv_perms[i]]], dim=0)
                    fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))
                    # print(f"fuse_logits: {fuse_logits}")

                else:
                    res = output['sequences'][:,:-1] == self.lang_model.cls_token_id[0] # last token is eos
                    res_check = torch.tensor([False] * batch_size, dtype=torch.bool)
                    hidden_states = torch.cat(output['hidden_states'],dim=1)
                    for i in range(batch_size):
                        res_check[i] = res[i].any()
                    if res_check.all():
                        predictions = self.out_head(
                            hidden_states[res])
                    else:
                        if self.args.action_first_in_gt:
                            predictions = self.out_head(output['hidden_states'][1].squeeze(1))
                        else:
                            predictions = self.out_head(output['hidden_states'][-1].squeeze(1))

               # outputs = {
                #     'fuse_embeds': fuse_embeds.detach(),
                #     "loss": loss
                # }

                # loss, hidden_states = output.loss, output.hidden_states

            if not multiple_sample_cot:
                for i in range(batch_size):
                    fuse_logits[i][cand_masks[i]] = torch.cat(
                        [predictions[i, 0:1], predictions[i, 1:cand_nums[i]][inv_perms[i]]], dim=0)

                fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))


            return {
                'fuse_embeds': fuse_embeds.detach(),
                'fuse_logits': fuse_logits,
                'logits':None,
                "generated_sentences_navigation_cot": generated_sentences,
                "action": action,
                "prompts": batch["prompts"],
                "new_cot": None,
                "cls_hidden_state": hidden_states[text_input['input_ids'] == self.lang_model.cls_token_id[0]] if self.args.enable_RL_A2C and not get_cot_output else None,
                "direction_landmark_dict":direction_landmark_dict if self.args.cot_summarization else None,
                "rand_perms": rand_perms if self.args.cot_summarization else None,
            }

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(4096, 768),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(768, 1),
        )
        # self.state2value = nn.Linear(4096, 1)

    def forward(self, state):
        output = self.state2value(state)
        print(f"before: output size {output.size()}")
        output = output.squeeze(1)
        print(f"after: output size {output.size()}")
        return output


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, layer_norm_eps=None):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, layer_norm_eps)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, layer_norm_eps=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self,config, layer_norm_eps=None):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(4096, config.hidden_size)
        # if isinstance(config.hidden_act, str):
        #     self.transform_act_fn = ACT2FN[config.hidden_act]
        # else:
        #ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.transform_act_fn = gelu
        #self.transform_act_fn = torch.nn.functional.relu
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
