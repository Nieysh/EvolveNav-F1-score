import math
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from models.ops import pad_tensors_wgrad
from models.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts
from .base_agent import BaseAgent

import numpy as np
import torch
from collections import defaultdict
from contextlib import nullcontext
from models.graph_utils import GraphMap
from typing import List
import spacy
import json
import os



def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device
        masks = torch.arange(max_len).to(device).repeat(len(seq_lens), 1) < seq_lens.unsqueeze(1)
        return masks

    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)

    seq_lens = np.array(seq_lens)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks

def get_results(pred_results, detailed_output=False):
    pred_output = []
    for k, v in pred_results.items():
        ret = {
            'instr_id': k,
            'trajectory': v['path']
        }

        # enable navigation cot
        if 'generated_sentences_navigation_cot' in v:
            ret.update({
                'generated_sentences_navigation_cot': v.get('generated_sentences_navigation_cot',''),
                'navigation_cot_gt': v.get('navigation_cot_gt', ''),
                'gmap_vpids': v.get('gmap_vpids',''),
                'gt_node': v.get('gt_node', '')
            })

        # scan_qa
        if 'answer' in v:
            ret.update({
                'pred_answer': v['generated_sentences'],
                'oracle_pred_answer': v.get('oracle_pred_answer', ''),
                'gt_answer': v['answer'],
            })

        # obj nav
        if 'pred_objid' in v:
            ret.update({
                'pred_objid': v['pred_objid'],
                'pred_obj_direction': v['pred_obj_direction']
            })
        pred_output.append(ret)

    return pred_output


class MP3DAgent(BaseAgent):
    def __init__(self, args, shortest_distances, shortest_paths):
        self.args = args
        self.shortest_paths = shortest_paths
        self.shortest_distances = shortest_distances
        self.nlp = spacy.load("en_core_web_lg")
        # buffer
        self.scanvp_cands = {}
        # newly added
        self.t2t_landmark_dir = str(args.data_dir/'t2t_landmarks')
        random.seed(args.seed)

    def update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

    def panorama_feature_variable_object(self, obs):
        ''' Extract precomputed features into variable. '''
        has_obj = 'obj_img_fts' in obs[0]

        batch_view_img_fts, batch_obj_img_fts, batch_loc_fts, batch_nav_types = [], [], [], []
        batch_view_lens, batch_obj_lens, batch_obj_loc_fts = [], [], []
        batch_cand_vpids, batch_objids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = torch.from_numpy(np.concatenate([view_ang_fts, view_box_fts], 1))

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))
            batch_loc_fts.append(view_loc_fts)

            # object
            if has_obj:
                batch_obj_loc_fts.append(torch.from_numpy(np.concatenate([ob['obj_ang_fts'], ob['obj_box_fts']], 1)))
                batch_objids.append(ob['obj_ids'])
                batch_obj_lens.append(len(ob['obj_img_fts']))
                batch_obj_img_fts.append(torch.from_numpy(ob['obj_img_fts']))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        ret = {
            'view_img_fts': batch_view_img_fts,
            'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types,
            'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

        if has_obj:
            batch_obj_img_fts = pad_tensors(batch_obj_img_fts).cuda()
            batch_obj_loc_fts = pad_tensors(batch_obj_loc_fts).cuda()
            batch_obj_lens = torch.LongTensor(batch_obj_lens).cuda()
            assert batch_obj_img_fts.shape[:2] == batch_obj_loc_fts.shape[
                                                  :2], f'shape of batch_obj_img_fts {batch_obj_img_fts.shape[:2]} must equal to shape of batch_obj_loc_fts {batch_obj_loc_fts.shape[:2]}'
            ret.update({
                'obj_img_fts': batch_obj_img_fts,
                'obj_loc_fts': batch_obj_loc_fts,
                'obj_lens': batch_obj_lens,
                'obj_ids': batch_objids,
            })

        return ret

    def panorama_feature_variable_12views(self, obs):
        batch_view_img_fts = []
        batch_loc_fts = []
        batch_view_lens = []
        batch_nav_types = []
        batch_cand_vpids = []

        for i, ob in enumerate(obs):
            view_img_fts = [x[:self.args.image_feat_size] for k, x in enumerate(ob['feature'])]
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = [x[self.args.image_feat_size:] for k, x in enumerate(ob['feature'])]
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_view_lens.append(len(view_img_fts))
            batch_nav_types.append(torch.LongTensor([1] * 12 + [0] * 24))
            batch_cand_vpids.append([None] * 36)

        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        ret = {
            "view_img_fts": batch_view_img_fts,
            "loc_fts": batch_loc_fts,
            "nav_types": batch_nav_types,
            "view_lens": batch_view_lens,
            "cand_vpids": batch_cand_vpids
        }
        return ret

    def get_direction_v2(self, current_idx, previous_idx, no_action=False, spatial_relation=False):
        # id2angle
        id2angle = {
            0: (0, -30), 1: (30, -30), 2: (60, -30), 3: (90, -30), 4: (120, -30), 5: (150, -30), 6: (180, -30),
            7: (210, -30), 8: (240, -30), 9: (270, -30), 10: (300, -30), 11: (330, -30), 12: (0, 0), 13: (30, 0),
            14: (60, 0), 15: (90, 0), 16: (120, 0), 17: (150, 0), 18: (180, 0), 19: (210, 0), 20: (240, 0),
            21: (270, 0),
            22: (300, 0), 23: (330, 0), 24: (0, 30), 25: (30, 30), 26: (60, 30), 27: (90, 30), 28: (120, 30),
            29: (150, 30),
            30: (180, 30), 31: (210, 30), 32: (240, 30), 33: (270, 30), 34: (300, 30), 35: (330, 30),
        }
        current_vp_angle = id2angle[current_idx]
        previous_vp_angle = id2angle[previous_idx]
        rel_heading = (current_vp_angle[0] - previous_vp_angle[0]) / 180 * math.pi
        rel_elevation = (current_vp_angle[1] - previous_vp_angle[1]) / 180 * math.pi
        # if rel_elevation > 0:
        #     direction_text = 'go up to'
        # elif rel_elevation < 0:
        #     direction_text = 'go down to'

        # A. front
        # B. rear
        # C. right
        # D. left
        # E. upon
        # F. under
        if no_action:
            action_direction_mapping = {
                'go forward to': "A. front",
                'go back to': "B. rear",
                'turn right to': "C. right",
                'turn left to': "D. left",
                'go up to': "E. upon",
                'go down to': "F. under"
            }

        if spatial_relation:
            action_spatial_relation_mapping = {
                'go forward to': "in front of",
                'go back to': "behind",
                'turn right to': "to the right of",
                'turn left to': "to the left of",
                'go up to': "above",
                'go down to': "below"
            }

        if current_vp_angle[1]-previous_vp_angle[1] > 0:
            direction_text = 'go up to'
        elif current_vp_angle[1]-previous_vp_angle[1] < 0:
            direction_text = 'go down to'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 4:
                    direction_text = 'go forward to'
                elif rel_heading < -math.pi / 4 and rel_heading >= -math.pi * 3 / 4:
                    direction_text = 'turn left to'
                elif rel_heading < -math.pi * 3 / 4 and rel_heading >= -math.pi * 5 / 4:
                    direction_text = 'go back to'
                elif rel_heading < -math.pi * 5 / 4 and rel_heading >= -math.pi * 7 / 4:
                    direction_text = 'turn right to'
                else:
                    direction_text = 'go forward to'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 4:
                    direction_text = 'go forward to'
                elif rel_heading > math.pi / 4 and rel_heading <= math.pi * 3 / 4:
                    direction_text = 'turn right to'
                elif rel_heading > math.pi * 3 / 4 and rel_heading <= math.pi * 5 / 4:
                    direction_text = 'go back to'
                elif rel_heading > math.pi * 5 / 4 and rel_heading <= math.pi * 7 / 4:
                    direction_text = 'turn left to'
                else:
                    direction_text = 'go forward to'
            elif rel_heading == 0:
                direction_text = 'go forward to'

        if no_action:
            direction_text = action_direction_mapping[direction_text]
        if spatial_relation:
            direction_text = action_spatial_relation_mapping[direction_text]

        return direction_text

    def remove_unwanted_landmarks(self, landmarks):
        ### TODO: landmark需要统计分析一下, room/hallway等。另外，全部处理完以后可视化出来看看
        unwanted_landmarks = ["floor", "inside", "wall", "ceiling", "house"]
        for i, item in enumerate(landmarks):
            for delete_landmark in unwanted_landmarks:
                if delete_landmark in item:
                    del landmarks[i]
                    break
        return landmarks

    # newly added:
    def panorama_feature_variable_sub_candidates(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        batch_QA_landmark_pairs, batch_QA_cand_GTs = [], []
        spatial_relation_flag = False

        for i, ob in enumerate(obs):
            cand_captions, cand_landmarks = [], []

            cand_viewidxs = []
            for j, cc in enumerate(ob['candidate']):
                # caption = ob['captions'][f"{ob['scan']}_{ob['viewpoint']}_{cc['pointId']}"]
                # caption = self.t2t_caption_data[f"{ob['scan']}_{ob['viewpoint']}_{cc['pointId']}"]
                # cand_captions.append(caption)
                # doc = self.nlp(caption)
                # landmarks = []
                # for noun in doc.noun_chunks:
                #     landmarks.append(noun.lemma_)
                # cand_landmarks.append(self.remove_unwanted_landmarks(landmarks)) # landmarks: landmarks of current candidate (cc)
                cand_landmarks.append(ob['landmarks'][f"{cc['pointId']}"])
                cand_viewidxs.append(cc['pointId'])

            # no_empty_cand_landmarks = [cand_landmarks[i] for i in range(len(cand_landmarks)) if len(cand_landmarks[i]) > 0]
            # no_empty_cand_viewidxs = [cand_viewidxs[i] for i in range(len(cand_landmarks)) if len(cand_landmarks[i]) > 0]
            # no_empty_cand_original_idx_in_cand = [i for i in range(len(cand_landmarks)) if len(cand_landmarks[i]) > 0]
            no_empty_cand_landmarks = [cand_landmarks[i] for i in range(len(cand_landmarks))]
            no_empty_cand_viewidxs = [cand_viewidxs[i] for i in range(len(cand_landmarks))]
            no_empty_cand_original_idx_in_cand = [i for i in range(len(cand_landmarks))]

            no_empty_new_cand_idx = list(range(len(no_empty_cand_landmarks)))

            QA_sub_cands = []  # record idx of no_empty cand_landmarks (idx of cand_landmarks & cand_viewidxs)
            QA_sub_cand_viewidxs = []
            QA_cur_cand = []
            batch_QA_sub_cand_original_idx = []

            query_cand = random.sample(no_empty_new_cand_idx, 1)[0]  # 可视化 query_cand
            query_landmarks = no_empty_cand_landmarks[query_cand]  # 可视化 query_landmarks
            QA_sub_cands.append([query_cand])
            QA_sub_cand_viewidxs.append([no_empty_cand_viewidxs[query_cand]])
            QA_cur_cand.append(obs[i]['viewIndex'])  # 可视化 obs[i]['viewIndex']
            batch_QA_landmark_pairs.append(query_landmarks)
            batch_QA_cand_GTs.append(
                self.get_direction_v2(no_empty_cand_viewidxs[query_cand], obs[i]['viewIndex']))  # 可视化 GT answer
            if self.args.all_cand_input:
                batch_QA_sub_cand_original_idx.append(no_empty_cand_original_idx_in_cand[query_cand]+1)
            else:
                batch_QA_sub_cand_original_idx.append(1)


            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            used_viewidxs = set()
            for i, per_QA_sub_cand_viewidxs in enumerate(QA_sub_cand_viewidxs):
                view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                     in enumerate(ob['feature']) if k == QA_cur_cand[i]])
                view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                     in enumerate(ob['feature']) if k == QA_cur_cand[i]])
                nav_types.append(1)
                cand_vpids.append(obs[i]['viewpoint'])

                if self.args.all_cand_input:
                    for j, cc in enumerate(ob['candidate']):
                        view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                        view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                        nav_types.append(1)
                        cand_vpids.append(cc['viewpointId'])
                        used_viewidxs.add(cc['pointId'])
                else:
                    for j, cc in enumerate(ob['candidate']):
                        if cc['pointId'] in per_QA_sub_cand_viewidxs and cc['pointId'] not in used_viewidxs:
                            view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                            view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                            nav_types.append(1)
                            cand_vpids.append(cc['viewpointId'])
                            used_viewidxs.add(cc['pointId'])

                # non cand views
                view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                     in enumerate(ob['feature']) if k not in used_viewidxs])
                view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                     in enumerate(ob['feature']) if k not in used_viewidxs])
                nav_types.extend([0] * (36 - len(used_viewidxs)))
                # combine cand views and noncand views
                view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
                view_ang_fts = np.stack(view_ang_fts, 0)
                view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
                view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

                batch_view_img_fts.append(torch.from_numpy(view_img_fts))
                batch_loc_fts.append(torch.from_numpy(view_loc_fts))
                batch_nav_types.append(torch.LongTensor(nav_types))
                batch_cand_vpids.append(cand_vpids)
                batch_view_lens.append(len(view_img_fts))

        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        # retry = 0
        # # randomly choose cand pairs for 3 times
        # while len(QA_sub_cands) < 3:
        #     if len(no_empty_new_cand_idx) >= 2:
        #         cand_pair = random.sample(no_empty_new_cand_idx, 2)
        #         if len(no_empty_new_cand_idx) > 2:
        #             disturb_cand = random.sample([i for i in no_empty_new_cand_idx if i not in cand_pair], 1)
        #         else:
        #             disturb_cand = None
        #
        #         # get ref_landmark and query_landmark for QA:
        #         ref_cand = cand_pair[0]
        #         query_cand = cand_pair[1]
        #         ref_possible_landmarks = [item for item in cand_landmarks[ref_cand] if item not in cand_landmarks[query_cand]]
        #         query_possible_landmarks = [item for item in cand_landmarks[query_cand] if item not in cand_landmarks[ref_cand]]
        #         if ref_possible_landmarks and query_possible_landmarks:
        #             ref_landmark = random.sample(ref_possible_landmarks,1)[0]
        #             query_landmark = random.sample(query_possible_landmarks,1)[0]
        #         else:
        #             ref_landmark = None
        #             query_landmark = None
        #         # check if ready
        #         if ref_landmark is not None and query_landmark is not None:
        #             if disturb_cand is not None:
        #                 QA_sub_cands.append(cand_pair + disturb_cand)
        #                 QA_sub_cand_viewidxs.append([cand_viewidxs[cand_pair[0]], cand_viewidxs[cand_pair[1]], cand_viewidxs[disturb_cand[0]]])
        #             else:
        #                 QA_sub_cands.append(cand_pair)
        #                 QA_sub_cand_viewidxs.append([cand_viewidxs[cand_pair[0]], cand_viewidxs[cand_pair[1]]])
        #             # QA_landmark_pair.append([ref_landmark, query_landmark])
        #             batch_QA_landmark_pairs.append([ref_landmark, query_landmark])
        #             batch_QA_cand_GTs.append(self.get_direction_v2(cand_viewidxs[query_cand], cand_viewidxs[ref_cand]))
        #         else:
        #             retry += 1
        #     elif (len(no_empty_new_cand_idx) == 1) or (retry >= 3):
        #         QA_sub_cands.append(no_empty_new_cand_idx[0])
        #         QA_sub_cand_viewidxs.append(cand_viewidxs[no_empty_new_cand_idx[0]])
        #         # QA_landmark_pair.append()
        #         batch_QA_landmark_pairs.append([cand_landmarks[no_empty_new_cand_idx[0]][0]])
        #         batch_QA_cand_GTs.append(self.get_direction_v2(cand_viewidxs[no_empty_new_cand_idx[0]],12))
        #     else:
        #         QA_sub_cands.append(None)
        #         QA_sub_cand_viewidxs.append(None)
        #         batch_QA_landmark_pairs.append(None)
        #         batch_QA_cand_GTs.append(None)

        # if QA_sub_cand_viewidxs:
        #     for per_QA_sub_cand_viewidxs in QA_sub_cand_viewidxs:
        #         view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
        #
        #         used_viewidxs = set()
        #         if per_QA_sub_cand_viewidxs is not None:
        #             # sub cand views
        #             for j, cc in enumerate(ob['candidate']):
        #                 if cc['pointId'] in per_QA_sub_cand_viewidxs and cc['pointId'] not in used_viewidxs:
        #                     view_img_fts.append(cc['feature'][:self.args.image_feat_size])
        #                     view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
        #                     nav_types.append(1)
        #                     cand_vpids.append(cc['viewpointId'])
        #                     used_viewidxs.add(cc['pointId'])
        #         else:
        #             for j, cc in enumerate(ob['candidate']):
        #                 view_img_fts.append(cc['feature'][:self.args.image_feat_size])
        #                 view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
        #                 nav_types.append(1)
        #                 cand_vpids.append(cc['viewpointId'])
        #                 used_viewidxs.add(cc['pointId'])
        #
        #         # non cand views
        #         view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
        #                              in enumerate(ob['feature']) if k not in used_viewidxs])
        #         view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
        #                              in enumerate(ob['feature']) if k not in used_viewidxs])
        #         nav_types.extend([0] * (36 - len(used_viewidxs)))
        #         # combine cand views and noncand views
        #         view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
        #         view_ang_fts = np.stack(view_ang_fts, 0)
        #         view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
        #         view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
        #
        #         batch_view_img_fts.append(torch.from_numpy(view_img_fts))
        #         batch_loc_fts.append(torch.from_numpy(view_loc_fts))
        #         batch_nav_types.append(torch.LongTensor(nav_types))
        #         batch_cand_vpids.append(cand_vpids)
        #         batch_view_lens.append(len(view_img_fts))
        #
        #     batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        #     batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        #     batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        #     batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        ### padding size: bs*1, seq_len, dim
        ### not padding: batch_QA_landmark_pairs (bs*1, 2 or 1)

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
            'QA_landmark_pairs': batch_QA_landmark_pairs,
            'QA_cand_GTs': batch_QA_cand_GTs,
            'QA_sub_cand_original_idx': batch_QA_sub_cand_original_idx
        }

    def get_pos_fts(self, cnt_vp, cand_vps, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in cand_vps:
            rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cnt_vp, vp,
                base_heading=cur_heading, base_elevation=cur_elevation,
            )
            rel_angles.append([rel_heading, rel_elevation])
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return rel_ang_fts

    def nav_vp_variable(self, obs, gmaps, pano_embeds, pano_masks, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )
        pano_masks = torch.cat(
            [torch.ones_like(pano_masks[:, :1]), pano_masks], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'pano_masks': pano_masks,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def nav_vp_variable_repeat(self, obs, gmaps, pano_embeds, pano_masks, cand_vpids, view_lens, nav_types, repeat=0):
        batch_size = len(obs)
        if repeat > 0:
            batch_size = batch_size * repeat
            gmaps = [gmap for gmap in gmaps for i in range(repeat)]

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )
        pano_masks = torch.cat(
            [torch.ones_like(pano_masks[:, :1]), pano_masks], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            ob_idx = int(i / repeat)
            # print(f"ob_idx:{ob_idx} i:{i} len(obs):{len(obs)}, len(cand_vpids):{len(cand_vpids)}")
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[ob_idx]['viewpoint'], cand_vpids[i],
                obs[ob_idx]['heading'], obs[ob_idx]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[ob_idx]['viewpoint'], [gmap.start_vp],
                obs[ob_idx]['heading'], obs[ob_idx]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'pano_masks': pano_masks,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if gmap.graph.visited(k):
                    visited_vpids.append(k)
                else:
                    unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )  # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def teacher_action_r4r(
            self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                is_r2r = 'r2r' in ob['instr_id']
                if imitation_learning and is_r2r:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0  # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:

                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0  # Stop if arrived
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    pass
                                    # dist = - cal_dtw(
                                    #     self.shortest_distances[scan],
                                    #     sum(traj[i]['path'], []) + self.shortest_paths[scan][ob['viewpoint']][vpid][1:],
                                    #     ob['gt_path'],
                                    #     threshold=3.0
                                    # )['nDTW']
                                elif self.args.expert_policy == 'spl':

                                    dist = self.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                           + self.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0  # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                   + self.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def teacher_object(self, obs):
        targets = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            i_vp = ob['viewpoint']
            i_objids = ob['obj_ids']
            if len(i_objids) == 0:
                targets[i] = self.args.ignoreid
            else:
                targets[i] = self.args.ignoreid  # target is not exist among the candidates
                if i_vp in ob['gt_end_vps']:
                    for j, obj_id in enumerate(i_objids):
                        if str(obj_id) == str(ob['gt_obj_id']):
                            targets[i] = j + 1
                            break
        return torch.from_numpy(targets).cuda()

    def prepare_cot(self, obs, feedback, data_type, nav_vpids, nav_targets, traj, gmaps, t):
        batch_navigation_cot_gt = []

        ### get navigation cot GT
        for i, ob in enumerate(obs):
            if nav_targets[i] == -100:
                nav_target = 0
            else:
                nav_target = nav_targets[i]

                # long-term goal
            if data_type[i] == 'r2r':
                if 'fg_instruction' in ob:
                    long_term_goal = ob['fg_instruction'][ob['fg_view'][-1]]
                    if ob['fg_view'][-1] != len(ob['fg_instruction']) - 1:
                        long_term_goal = ', '.join([long_term_goal, ob['fg_instruction'][-1]]) + '.'
                    else:
                        long_term_goal = ob['fg_instruction'][ob['fg_view'][-1]] + '.'
                else:
                    long_term_goal = ob['instruction']
            if data_type[i] == 'reverie':
                long_term_goal = ob['instruction']
            long_term_goal = f"- Long-term Goal: {long_term_goal}"

            # short-term goal
            if data_type[i] == 'r2r':
                if feedback == 'teacher' and 'fg_instruction' in ob:
                        if t >= len(ob['fg_instruction']):
                            t = -1
                        short_term_goal = ob['fg_instruction'][t]
                if feedback == 'sample' or 'fg_instruction' not in ob:
                    gt_vpid = nav_vpids[i][nav_target]
                    if gt_vpid is not None:
                        gt_sub_path = gmaps[i].graph.path(ob['viewpoint'], gt_vpid)
                        if len(gt_sub_path) == 1:
                            prev_vp = traj[i]['path'][-1][-1]
                        else:
                            prev_vp = gt_sub_path[-2]
                        viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][gt_vpid]
                        gt_landmarks = load_json(
                            os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid, f"{ob['scan']}_{gt_vpid}.json"))[str(viewidx)]
                        short_term_goal = f"Go to the direction of [{', '.join(gt_landmarks)}]"
                    else:
                        short_term_goal = "Stop"
            if data_type[i] == 'reverie':
                gt_vpid = nav_vpids[i][nav_target]
                if gt_vpid is not None:
                    traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], gt_vpid))
                    gt_sub_path = gmaps[i].graph.path(ob['viewpoint'], gt_vpid)
                    if len(gt_sub_path) == 1:
                        prev_vp = traj[i]['path'][-1][-1]
                    else:
                        prev_vp = gt_sub_path[-2]
                    viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][gt_vpid]
                    gt_landmarks = load_json(
                        os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid, f"{ob['scan']}_{gt_vpid}.json"))[str(viewidx)]
                    short_term_goal = f"Go to the direction of [{', '.join(gt_landmarks)}]"
                else:
                    short_term_goal = "Stop"
            short_term_goal = f"- Short-term Goal: {short_term_goal}."

            # spatial relation
            if data_type[i] == 'r2r' or data_type[i] == 'reverie':
                cand_viewidxs, cand_position = [], []
                cand_spatial_relation = []
                for x, vpid in enumerate(nav_vpids[i]):
                    found_cc = False
                    if x == 0:
                        cand_spatial_relation.append(f"Cand ({x}) means to stop.")
                    else:
                        for j, cc in enumerate(ob['candidate']):
                            if cc['viewpointId'] == vpid:
                                cand_viewidxs.append(cc['pointId'])
                                cand_spatial_relation.append(f"Cand ({x}) shows [{', '.join(ob['landmarks'][str(cc['pointId'])])}] {self.get_direction_v2(cc['pointId'], obs[i]['viewIndex'], spatial_relation=True)} me.")
                                found_cc = True
                        # if not found_cc:
                        #     cand_spatial_relation.append(f"Cand ({x}) means to go back to the previous unexplored direction.")
                spatial_relation = ' '.join(cand_spatial_relation)
            spatial_relation = f"- Spatial Relation: {spatial_relation}"

            # common sense
            if data_type[i] == 'r2r' or data_type[i] == 'reverie':
                gt_vpid = nav_vpids[i][nav_target]
                if gt_vpid is not None:
                    gt_sub_path = gmaps[i].graph.path(ob['viewpoint'], gt_vpid)
                    if len(gt_sub_path) == 1:
                        prev_vp = traj[i]['path'][-1][-1]
                    else:
                        prev_vp = gt_sub_path[-2]
                    viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][gt_vpid]
                    gt_landmarks = load_json(
                        os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid, f"{ob['scan']}_{gt_vpid}.json"))[
                        str(viewidx)]
                    common_sense = f"An observation with [{', '.join(gt_landmarks)}] may match with short-term goal and is likely to lead to the long-term goal."
                else:
                    common_sense = f"Observation matches with long-term goal, so stop."
                common_sense = f"- Reasoning: {common_sense}"

            navigation_cot_components = [long_term_goal, short_term_goal, common_sense] # TODO: spatial relation
            navigation_cot_gt = '\n'.join(navigation_cot_components)
            batch_navigation_cot_gt.append(navigation_cot_gt)
            # print(f"batch_navigation_cot_gt:{batch_navigation_cot_gt}")
        return batch_navigation_cot_gt

    def make_equiv_action(self, a_t, gmaps, obs, traj=None, env=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:  # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                env[i].sims[0].newEpisode([ob['scan']], [action], [heading], [elevation])

    def train(
            self,
            name,
            batch,
            args,
            config,
            model,
            criterion,
            dataset,
            step=0,
            entropy_metric=None,
            instr_pred_metric=None,
            **kwargs
    ):
        dataset_cfg = config.Pretrain if args.stage == 'pretrain' else config.Multi
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)

        if args.only_cot_training:
            if args.self_improving_cot:
                loss, _, reasoning_output = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                )
            else:
                loss, _ = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                )
        else:
            if args.stage == 'pretrain' or step % 2 == 0:
                #################### imitation learning ####################
                loss, _ = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                )
                if args.train_with_twice_forward_gt_and_selfoutput:
                    print(f"train w gt cot loss:{loss}")
                    loss2, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,
                        enable_self_cot=True
                    )
                    print(f"train w self cot loss:{loss2}")
                    loss += loss2
            else:
                #################### dagger training ####################
                loss, _ = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="sample", train_ml=loss_coef,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                )
                # if args.train_with_twice_forward_gt_and_selfoutput:
                #     print(f"train w gt cot loss:{loss}")
                #     loss2, _ = self.rollout(
                #         args, name, config.Optim, batch,
                #         model=model, criterion=criterion, dataset=dataset,
                #         feedback="sample", train_ml=loss_coef,
                #         entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,
                #         enable_self_cot=True
                #     )
                #     print(f"train w self cot loss:{loss2}")
                #
                #     loss += loss2

        return loss * args.gradient_accumulation_step

    def validate(
            self,
            name,
            args,
            config,
            model,
            loader,
            entropy_metric=None,
            instr_pred_metric=None,
    ):
        results = {}
        trie = None
        looped = False
        dataset = loader.get_dataset()
        pbar = tqdm(loader, disable=args.rank != 0)
        if name in ['EQA']:
            if hasattr(model, 'module'):
                tokenizer = model.module.lang_model.tokenizer
            else:
                tokenizer = model.lang_model.tokenizer

            trie = Trie(tokenizer.bos_token_id, tokenizer.eos_token_id)
            for word in dataset.answer_vocab:
                token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                if isinstance(tokenizer, LlamaTokenizer):
                    token_ids = [tokenizer.bos_token_id] + token_ids
                trie.insert(token_ids)

        for i, batch in enumerate(pbar):
            ml_loss, traj = self.rollout(
                args, name, config.Optim, batch,
                model=model, criterion=None, dataset=dataset,
                feedback="sample" if args.do_sample else "argmax", train_ml=None,
                entropy_metric=entropy_metric, instr_pred_metric=None,
                validate=True, trie=trie
            )

            for s_traj in traj:
                if s_traj['instr_id'] in results:
                    looped = True
                else:
                    ml_loss = 0
                    results[s_traj['instr_id']] = s_traj

            # Caldulate oracle prediction answer
            if name in ["EQA"]:
                _, oracle_traj = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=None, dataset=dataset,
                    feedback="teacher", train_ml=1,
                    entropy_metric=entropy_metric, instr_pred_metric=None,
                    validate=True, trie=trie
                )
                for s_traj in oracle_traj:
                    results[s_traj['instr_id']]['oracle_pred_answer'] = s_traj['generated_sentences']

            if looped:
                break

        preds = get_results(results)
        return preds

    def rollout(
            self,
            args,
            name,
            config,
            batch_dict,
            model,
            criterion,
            dataset,
            feedback,
            train_ml,
            entropy_metric,
            instr_pred_metric,
            validate=False,
            enable_self_cot=False,
            **kwargs
    ):
        """
        :param args:
        :param name: task name
        :param config:
        :param batch_dict:
        :param model:
        :param criterion:
        :param dataset:
        :param feedback:
        :param train_ml:
        :param entropy_metric:
        :param validate:
        :return:
        """
        obs = batch_dict["observations"]
        envs = batch_dict["env"]
        data_type = batch_dict['data_type']

        max_action_len = config.val_max_action_len[name] if validate else config.train_max_action_len[name]

        self.update_scanvp_cands(obs)
        batch_size = len(obs)

        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        if args.enable_navigation_cot:
            traj = [{
                'instr_id': ob['instr_id'],
                'path': [[ob['viewpoint']]],
                'scan': ob['scan'],
                'generated_sentences_navigation_cot':{},
                'navigation_cot_gt': {},
                'gmap_vpids': {},
                'gt_node': {},
                'details': {},
            } for ob in obs]
        else:
            traj = [{
                'instr_id': ob['instr_id'],
                'path': [[ob['viewpoint']]],
                'details': {},
            } for ob in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # instructions = language_variable(obs, data_type=batch_dict['data_type'])
        instructions = [ob["instruction"] for ob in obs]

        history = []
        hist_vis = []
        for idx in range(len(instructions)):
            history.append([])
            hist_vis.append([])

        entropys = []
        ml_loss, cnt_loss = 0., 0.
        flag = False

        ##newly added
        vp_landmarks = [{} for ob in obs]

        for t in range(max_action_len):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # multi-gpu
                if ended.all() or t == max_action_len - 1:
                    flag = True
                    context = nullcontext
                else:
                    context = model.no_sync
            else:
                # single-gpu
                if ended.all() or t == max_action_len - 1:
                    flag = True
                    context = nullcontext
                else:
                    context = nullcontext

            with context():

                for i, gmap in enumerate(gmaps):
                    if not ended[i]:
                        gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

                # graph representation
                pano_inputs = self.panorama_feature_variable_object(obs)
                panorama_out = model('panorama', pano_inputs)
                pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                  torch.sum(pano_masks, 1, keepdim=True)  # [B, D=768]

                for i, gmap in enumerate(gmaps):
                    if not ended[i]:
                        # update visited node
                        i_vp = obs[i]['viewpoint']
                        update_avg_pana_embeds = avg_pano_embeds[i].detach()  # update average features for gmap.
                        gmap.update_node_embed(i_vp, update_avg_pana_embeds, rewrite=True)
                        # update unvisited nodes
                        for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                            if not gmap.graph.visited(i_cand_vp):
                                update_pano_embeds = pano_embeds[i, j].detach()
                                gmap.update_node_embed(i_cand_vp, update_pano_embeds)

                # navigation policy
                nav_inputs = self.nav_gmap_variable(obs, gmaps)
                nav_inputs.update(
                    self.nav_vp_variable(
                        obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                        pano_inputs['view_lens'], pano_inputs['nav_types'],
                    )
                )

                nav_inputs.update({
                    'view_lens': pano_inputs['view_lens'],
                    'instruction': instructions,
                    'history': history,
                    'hist_vis': hist_vis,
                    'data_type': data_type
                })

                for i, ob in enumerate(obs):
                    for j, cc in enumerate(ob['candidate']):
                        if cc['viewpointId'] not in vp_landmarks[i]:
                            vp_landmarks[i][cc['viewpointId']]=ob['landmarks'][f"{cc['pointId']}"]
                nav_inputs.update({
                    'vp_landmarks': vp_landmarks
                })

                in_progress = torch.tensor(ended).logical_not()
                if ended.all():
                    in_progress[0] = True

                nav_vpids = nav_inputs['gmap_vpids']
                imitation_learning = feedback == 'teacher'
                # # Imitation Learning
                # if train_ml is not None:
                # [1] Supervised training
                if 'r2r' in data_type:
                    nav_targets = self.teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'],
                        imitation_learning=imitation_learning, t=t, traj=traj
                    )
                else:
                    nav_targets = self.teacher_action(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'],
                    )

                # newly added: navigation cot
                if data_type[0] in ['r2r', 'soon', 'reverie', 'r2r_aug', 'reverie_aug']:
                    enable_navigation_cot = (feedback == 'teacher' or feedback == 'sample' or feedback == 'argmax') and args.enable_navigation_cot
                    # and (not flag)
                if enable_navigation_cot:
                    nav_inputs.update({
                        "navigation_cot_gt": self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                                                              nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                                                              gmaps=gmaps, t=t)})
                    # print(f"nav_inputs['navigation_cot_gt']: {nav_inputs['navigation_cot_gt']}")
                    nav_inputs["prompts"] = self.prepare_prompts(
                        "navigation_cot",
                        nav_inputs
                    )
                    if self.args.check_cot_input_gt:
                        print(f"inputs: {nav_inputs['prompts']} gt: {nav_inputs['navigation_cot_gt']}")

                    if self.args.random_train_with_self_output_cot:
                        random_train_with_self_output = (t % 2 == 0)
                        assert self.args.train_with_self_output_cot == False, f"set both 'random_train_with_self_output_cot' and 'train_with_self_output' to True, random is disabled"
                    else:
                        random_train_with_self_output = False

                    if self.args.test_with_cot_gt and validate:
                        batch_generated_sentences_navigation_cot = []
                    else:
                        output = model('navigation_cot', nav_inputs, training=not validate, **kwargs)
                        if not validate:
                            lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                            lm_loss.backward()
                            # print(f"{t} cot ntp loss backward successfully")
                            instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                            ml_loss += lm_loss.detach()

                            if self.args.train_with_self_output_cot or self.args.random_train_with_self_output_cot or enable_self_cot:
                                self_cot_output = model('navigation_cot', nav_inputs, training=False, **kwargs)
                                batch_generated_sentences_navigation_cot = self_cot_output["generated_sentences_navigation_cot"]

                            if self.args.self_improving_cot:

                                reasonings = []
                                output_reasoning = {}
                                batch_output_reasoning = []
                                for _ in range(3):
                                    reasonings.append(model('navigation_cot', nav_inputs, training=False, do_sample=True, temperature=0.5))
                                for i in range(batch_size):
                                    pos_cot = []
                                    neg_cot = []
                                    pos_cot.append(nav_inputs['navigation_cot_gt'][i])
                                    for j in range(3):
                                        # TODO: check whether cot is good or bad
                                        pos = check_cot(reasonings[j][i])
                                        if pos:
                                            pos_cot.appned(reasonings[j][i])
                                        else:
                                            neg_cot.append(reasonings[j][i])
                                    if pos_cot:
                                        output_reasoning[obs[i]['instr_id']]['SFT'] = random.sample(pos_cot,1)
                                        if neg_cot:
                                            output_reasoning[obs[i]['instr_id']]['REF'] = (random.sample(pos_cot,1), random.sample(neg_cot,1))
                                            output_reasoning[obs[i]['instr_id']]['SEL'] = (random.sample(pos_cot,1), random.sample(neg_cot,1))
                                        else:
                                            output_reasoning[obs[i]['instr_id']]['REF'] = None
                                            output_reasoning[obs[i]['instr_id']]['SEL'] = None
                                    else:
                                        output_reasoning[obs[i]['instr_id']]['SFT'] = None
                                    batch_output_reasoning.append(output_reasoning)


                        else:
                            batch_generated_sentences_navigation_cot = []
                            for i in range(batch_size):
                                traj[i]['generated_sentences_navigation_cot'][t] = output["generated_sentences_navigation_cot"][i]
                                traj[i]['navigation_cot_gt'][t] = nav_inputs['navigation_cot_gt'][i]
                                traj[i]['gmap_vpids'][t] = nav_vpids[i]
                                traj[i]['gt_node'][t] = nav_targets[i].item()
                            batch_generated_sentences_navigation_cot = output["generated_sentences_navigation_cot"]
                            print(f"batch_generated_sentences_navigation_cot:{batch_generated_sentences_navigation_cot}")

                if self.args.only_cot_training:
                    imitation_learning = feedback == 'teacher'
                    # Imitation Learning
                    if train_ml is not None:
                        # [1] Supervised training
                        if 'r2r' in data_type:
                            nav_targets = self.teacher_action_r4r(
                                obs, nav_vpids, ended,
                                visited_masks=nav_inputs['gmap_visited_masks'],
                                imitation_learning=imitation_learning, t=t, traj=traj
                            )
                        else:
                            nav_targets = self.teacher_action(
                                obs, nav_vpids, ended,
                                visited_masks=nav_inputs['gmap_visited_masks'],
                            )
                else:
                    # graph representation
                    pano_inputs = self.panorama_feature_variable_object(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']

                    # navigation policy
                    # nav_inputs.update(self.nav_gmap_variable(obs, gmaps))
                    nav_inputs.update(self.nav_gmap_variable(obs, gmaps))
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )

                    nav_inputs.update({
                        'view_lens': pano_inputs['view_lens'],
                        'instruction': instructions,
                        'history': history,
                        'hist_vis': hist_vis,
                        'data_type': data_type
                    })

                    if enable_navigation_cot:
                        navigation_mode = "navigation_cot_decision"
                        nav_inputs.update({
                            "nav_cot": nav_inputs["navigation_cot_gt"] if not validate else batch_generated_sentences_navigation_cot
                        })
                        if self.args.test_with_cot_gt and validate:
                            nav_inputs["nav_cot"] = nav_inputs["navigation_cot_gt"]
                        if (self.args.train_with_self_output_cot or random_train_with_self_output or enable_self_cot) and not validate:
                            nav_inputs["nav_cot"] = batch_generated_sentences_navigation_cot
                    else:
                        navigation_mode = "navigation"

                    nav_inputs["prompts"] = self.prepare_prompts(
                        navigation_mode,
                        nav_inputs,
                        cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                        model.lang_model.cls_token[0]
                    )
                    # if enable_self_cot:
                    #     print(f"enable_self_cot: {enable_self_cot} nav_inputs['prompts']:{nav_inputs['prompts']}")
                    nav_outs = model('navigation', nav_inputs)

                    # dynamic fusion
                    nav_logits = nav_outs['fuse_logits']

                    nav_probs = torch.softmax(nav_logits / args.temperature, 1)

                    imitation_learning = feedback == 'teacher'
                    # Imitation Learning
                    if train_ml is not None:
                        # [1] Supervised training
                        if 'r2r' in data_type:
                            nav_targets = self.teacher_action_r4r(
                                obs, nav_vpids, ended,
                                visited_masks=nav_inputs['gmap_visited_masks'],
                                imitation_learning=imitation_learning, t=t, traj=traj
                            )
                        else:
                            nav_targets = self.teacher_action(
                                obs, nav_vpids, ended,
                                visited_masks=nav_inputs['gmap_visited_masks'],
                            )
                        ############# Single-Step Loss #############
                        cnt_loss += criterion(nav_logits,
                                              nav_targets) * train_ml / batch_size / args.gradient_accumulation_step

                        ml_loss += cnt_loss.detach()

                        ########### Single-Step Backward ###########
                        if not validate:
                            cnt_loss.backward()
                        cnt_loss = 0.

                if feedback == 'teacher':  # imitation learning
                    a_t = nav_targets  # teacher forcing
                elif feedback == 'sample':
                    c = torch.distributions.Categorical(nav_probs.float())
                    entropy_metric.accumulate(c.entropy().sum().item() / batch_size)  # For log
                    entropys.append(c.entropy())  # For optimization
                    a_t = c.sample().detach()
                elif feedback == 'argmax':
                    _, a_t = nav_logits.max(1)  # student forcing - argmax
                    a_t = a_t.detach()
                else:
                    print(feedback)
                    raise NotImplementedError

                for idx in range(len(a_t)):
                    if a_t[idx] == -100:
                        continue
                    history[idx] += ['<hist>']
                    if args.only_cot_training:
                        hist_vis[idx].append(output['fuse_embeds'][idx][a_t[idx]])
                    else:
                        hist_vis[idx].append(nav_outs['fuse_embeds'][idx][a_t[idx]])
                    if args.add_cand_landmark and len(hist_vis[idx])>=6:
                        hist_vis[idx] = hist_vis[idx][len(hist_vis[idx])-6:]
                        history[idx] = history[idx][len(history[idx])-6:]

                if not validate:
                    # if feedback == 'teacher' or feedback == 'sample':  # in training
                    assert feedback in ['teacher',
                                        'sample'], "Feedback must be either `teacher' or `sample' in training. "
                    a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
                else:
                    a_t_stop = a_t == 0

                ########### Object Prediction Sub-task ###########
                if (data_type[0] in ['soon', 'reverie']) and args.enable_og and flag:
                    # graph representation
                    pano_inputs = self.panorama_feature_variable_object(obs)
                    panorama_out = model('panorama', pano_inputs)

                    if 'obj_embeds' not in panorama_out:
                        pano_embeds = panorama_out['pano_embeds']
                        panorama_out.update({
                            "obj_embeds": torch.zeros((pano_embeds.shape[0], 0, pano_embeds.shape[2]),
                                                      dtype=pano_embeds.dtype, device=pano_embeds.device),
                            "obj_masks": torch.zeros((pano_embeds.shape[0], 0), dtype=torch.int64,
                                                     device=pano_embeds.device),
                            "obj_loc_fts": torch.zeros((pano_embeds.shape[0], 0, 7), dtype=pano_embeds.dtype,
                                                       device=pano_embeds.device)
                        })

                    nav_inputs.update({
                        'obj_embeds': panorama_out['obj_embeds'],
                        'obj_masks': panorama_out['obj_masks'],
                        'obj_loc_fts': panorama_out['obj_loc_fts']
                    })

                    nav_inputs.update({
                        'view_lens': pano_inputs['view_lens'],
                        'instruction': instructions,
                        'history': history,
                        'hist_vis': hist_vis,
                        'data_type': data_type
                    })
                    nav_inputs["prompts"] = self.prepare_prompts(
                        "object_grounding",
                        nav_inputs,
                        cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                        model.lang_model.cls_token[0]
                    )
                    obj_logits = model('object_grounding', nav_inputs)['obj_logits']
                    obj_targets = self.teacher_object(obs)

                    if not validate:
                        obj_loss = criterion(obj_logits,
                                             obj_targets) * args.obj_loss_coef / batch_size / args.gradient_accumulation_step
                        obj_loss.backward()
                        ml_loss += obj_loss.detach()

                    # update obj results
                    for i, gmap in enumerate(gmaps):
                        i_vp = obs[i]['viewpoint']
                        i_objids = obs[i]['obj_ids']
                        i_obj_logits = obj_logits[i, 1:]
                        if 'obj_directions' in obs[i]:
                            traj[i].update({
                                'pred_objid': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                                'pred_obj_direction': obs[i]['obj_directions'][torch.argmax(i_obj_logits)] if len(
                                    i_objids) > 0 else None,
                            })
                        else:
                            traj[i].update({
                                'pred_objid': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                                'pred_obj_direction': None,
                            })

                ########### Fine-grained R2R Sub-task ###########
                enable_fgr2r = (feedback == 'teacher') and (not flag) and (not a_t_stop[0]) and (data_type[0] == 'r2r') and (not validate) and ('fg_instruction' in ob) and ('fg_view' in ob) and args.enable_fgr2r
                if enable_fgr2r:
                    pano_inputs = self.panorama_feature_variable_12views(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )
                    nav_inputs['instruction'] = ['where are we going with direction ({}) ?'.format(idx) for idx in
                                                 nav_targets]
                    nav_inputs["data_type"] = ['fgr2r' for idx in nav_targets]
                    nav_inputs['answer'] = [ob['fg_instruction'][ob['fg_view'][t]] if t<=len(ob['fg_view'])-1 else ob['fg_instruction'][ob['fg_view'][-1]] for ob in obs]
                    nav_inputs['hist_vis'] = [[] for idx in nav_targets]
                    nav_inputs['history'] = [[] for idx in nav_targets]
                    nav_inputs["prompts"] = self.prepare_prompts("embodied_qa", nav_inputs)
                    output = model('embodied_qa', nav_inputs, training=not validate, **kwargs)
                    if not validate:
                        lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                        lm_loss.backward()
                        instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                        ml_loss += lm_loss.detach()

                ########### Navigation Summarization Sub-task ###########
                if data_type[0] == 'eqa':
                    enable_summarize = flag
                elif data_type[0] in ['r2r', 'soon', 'reverie', 'r2r_aug', 'reverie_aug']:
                    enable_summarize = (feedback == 'teacher' or feedback == 'argmax') and flag and args.enable_summarize and (
                                                   not validate or args.mode == 'test')
                elif data_type[0] in ['cvdn']:
                    enable_summarize = False
                else:
                    raise NotImplementedError

                if enable_summarize:  # gen loss

                    pano_inputs = self.panorama_feature_variable_12views(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )

                    nav_inputs['instruction'] = [ob["instruction"] for ob in obs]
                    nav_inputs['history'] = history
                    nav_inputs['hist_vis'] = hist_vis
                    nav_inputs["data_type"] = data_type
                    nav_inputs['answer'] = [ob.get('answer', '') for ob in obs]
                    nav_inputs["prompts"] = self.prepare_prompts("summarization", nav_inputs)
                    output = model('summarization', nav_inputs, training=not validate, **kwargs)
                    if not validate:
                        lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                        lm_loss.backward()
                        instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                        ml_loss += lm_loss.detach()
                    else:
                        for i in range(batch_size):
                            generated_sentences = output["generated_sentences"]
                            traj[i]['generated_sentences'] = generated_sentences[i]
                            traj[i]['answer'] = nav_inputs['answer'][i]

                ########### Navigation Spatial Relation Sub-task ###########
                if data_type[0] in ['r2r', 'soon', 'reverie', 'r2r_aug', 'reverie_aug']:
                    enable_spatial_relation = (feedback == 'teacher' or feedback == 'argmax') and args.enable_spatial_relation and (
                                                          not validate or args.mode == 'test')
                if enable_spatial_relation:
                    ### TODO:Check
                    ### input
                    pano_inputs = self.panorama_feature_variable_sub_candidates(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable_repeat(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'], repeat=pano_embeds.size(0)
                        )
                    )
                    nav_inputs.update(
                        {"QA_landmark": pano_inputs["QA_landmark_pairs"],
                         "QA_cand_GTs": pano_inputs["QA_cand_GTs"],
                         "QA_sub_cand_original_idx": pano_inputs["QA_sub_cand_original_idx"], })
                    nav_inputs["data_type"] = data_type
                    nav_inputs['instruction'] = [ob["instruction"] for ob in obs]
                    nav_inputs["prompts"] = self.prepare_prompts("spatial_relation", nav_inputs)  # 可视化 prompt

                    output = model('spatial_relation', nav_inputs, training=not validate, **kwargs)
                    if not validate:
                        lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                        lm_loss.backward()
                        instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                        ml_loss += lm_loss.detach()
                    else:
                        for i in range(batch_size):
                            generated_sentences_spatial_relation = output["generated_sentences_spatial_relation"]
                            traj[i]['generated_sentences_spatial_relation'] = generated_sentences_spatial_relation[i]
                            traj[i]['spatial_relation_answer'] = nav_inputs['QA_cand_GTs'][i]

                # if enable_common_sense:
                ### TODO
                # continue

                # if enable_action:
                ### TODO
                # continue

                # Prepare environment action
                cpu_a_t = []
                for i in range(batch_size):
                    # TODO
                    if False and data_type[i] == 'eqa':
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == max_action_len - 1):
                            cpu_a_t.append(None)
                            just_ended[i] = True
                        else:
                            cpu_a_t.append(nav_vpids[i][a_t[i]])

                # Make action and get the new state
                self.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)

                for i in range(batch_size):
                    if (not ended[i]) and just_ended[i]:
                        stop_node, stop_score = None, {'stop': -float('inf')}
                        for k, v in gmaps[i].node_stop_scores.items():
                            if v['stop'] > stop_score['stop']:
                                stop_score = v
                                stop_node = k
                        if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                            traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))

                # get new observation and update graph
                new_obs = []
                for b_i in range(batch_size):
                    # TODO
                    if False and data_type[b_i] == 'eqa':
                        raise NotImplementedError
                    else:
                        new_obs.append(
                            dataset.get_obs(
                                items=[batch_dict['item'][b_i]],
                                env=envs[b_i], data_type=data_type[b_i]
                            )[0]
                        )
                obs = new_obs

                self.update_scanvp_cands(obs)

                for i, ob in enumerate(obs):
                    if not ended[i]:
                        gmaps[i].update_graph(ob)

                ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

                if flag:
                    break

        if self.args.self_improving_cot:
            return ml_loss, traj, batch_output_reasoning
        else:
            return ml_loss, traj

    def prepare_prompts(self, mode, batch, **kwargs):
        batch_size = len(batch["instruction"])
        if mode == "navigation":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cls_token=kwargs.get("cls_token"),
                    )
                )
        elif mode == "summarization" or mode == "embodied_qa":
            hist_nums = [len(his) for his in batch["history"]]
            vp_nav_masks = batch["vp_nav_masks"][:, 1:]
            cand_nums = vp_nav_masks.sum(1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        mode,
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                    )
                )
        elif mode == "object_grounding":
            hist_nums = [len(his) for his in batch["history"]]
            cand_nums = batch["obj_masks"].sum(dim=1) + 1    # add not exist
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        mode,
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cls_token=kwargs.get("cls_token"),
                    )
                )
        elif mode == "spatial_relation":
            batch_size = len(batch["QA_landmark"])

            prompts = []
            vp_nav_masks = batch["vp_nav_masks"][:, 1:]
            cand_nums = vp_nav_masks.sum(1)
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        mode,
                        cand_num=cand_nums[bn],
                        QA_landmark=batch["QA_landmark"][bn],
                        query_cand_id=batch["QA_sub_cand_original_idx"][bn]
                    )
                )
        elif mode == "navigation_cot":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation_cot",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cand_landmarks=batch["vp_landmarks"][bn],
                        nav_vpids=batch['gmap_vpids'][bn],
                        cand_masks=cand_masks[bn],
                    )
                )
        elif mode == "navigation_cot_decision":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation_cot_decision",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cls_token=kwargs.get("cls_token"),
                        cot_input=batch["nav_cot"][bn],
                        cand_landmarks=batch["vp_landmarks"][bn],
                        nav_vpids=batch['gmap_vpids'][bn],
                        cand_masks=cand_masks[bn],
                    )
                )
        else:
            raise NotImplementedError

        return prompts