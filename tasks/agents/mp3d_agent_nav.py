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
            gmap_img_embeds 