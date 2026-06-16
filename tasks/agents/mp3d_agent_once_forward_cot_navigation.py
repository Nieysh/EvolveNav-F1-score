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
# import spacy
import json
import os
import collections
import torch.nn as nn
import re
from tools.parser import random_seed

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
            'trajectory': v['path'],
            'scan': v.get('scan', ''),
        }

        # enable navigation cot
        if 'generated_sentences_navigation_cot' in v:
            ret.update({
                'generated_sentences_navigation_cot': v.get('generated_sentences_navigation_cot',''),
                'navigation_cot_gt': v.get('navigation_cot_gt', ''),
                'gmap_vpids': v.get('gmap_vpids',''),
                'gt_node': v.get('gt_node', ''),
                'pred_node': v.get('pred_node', ''),
                'gt_vpid': v.get('gt_vpid', ''),
                'pred_vpid': v.get('pred_vpid', ''),
                'gt_action_viewpoint': v.get('gt_action_viewpoint', ''),
                'gt_viewidx': v.get('gt_viewidx', ''),
                'gt_landmarks': v.get('gt_landmarks', ''),
                'direction_of_gt': v.get('direction_of_gt', ''),
                'prompts': v.get('prompts', ''),
                'navigable_gmap_vpids': v.get('navigable_gmap_vpids', ''),
                'cot_decision_consistency': v.get('cot_decision_consistency', ''),
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

def remove_article(item):
    if item[:2] == 'a ' or item[:3] == 'an ' or item[:4] == 'the ':
        item = ' '.join(item.split(' ')[1:])
    return item


class MP3DAgent(BaseAgent):
    def __init__(self, args, shortest_distances, shortest_paths):
        self.args = args
        self.shortest_paths = shortest_paths
        self.shortest_distances = shortest_distances
        # self.nlp = spacy.load("en_core_web_lg")
        # buffer
        self.scanvp_cands = {}
        # newly added
        self.t2t_landmark_dir = str(args.data_dir/'t2t_landmarks')
        random.seed(args.seed)
        #random_seed(args.seed+args.rank)

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

    def get_direction_v2(self, current_idx, previous_idx, no_action=False, spatial_relation=False, spatial_answer=False):
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

        if spatial_answer:
            action_spatial_relation_mapping = {
                'go forward to': "front",
                'go back to': "rear",
                'turn right to': "right",
                'turn left to': "left",
                'go up to': "upon",
                'go down to': "under"
            }

        if current_vp_angle[1] > 0:
            direction_text = 'go up to'
        elif current_vp_angle[1] < 0:
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
        if spatial_answer:
            direction_text = action_spatial_relation_mapping[direction_text]

        return direction_text

    def get_direction_vp(self, cnt_vp, vp, cur_heading, cur_elevation, no_action=False, spatial_relation=False):
        rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cnt_vp, vp,
                base_heading=cur_heading, base_elevation=0
            )

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
            if self.args.mlm:
                action_spatial_relation_mapping = {
                    'go forward to': "go forward",
                    'go back to': "go back",
                    'turn right to': "turn right",
                    'turn left to': "turn left",
                    'go up to': "go up",
                    'go down to': "go down",
                }
            else:
                action_spatial_relation_mapping = {
                    'go forward to': "in front of",
                    'go back to': "behind",
                    'turn right to': "to the right of",
                    'turn left to': "to the left of",
                    'go up to': "above",
                    'go down to': "below"
                }

        rel_elevation = math.degrees(rel_elevation)

        if rel_elevation > 10:
            direction_text = 'go up to'
        elif rel_elevation < -10:
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
                self.get_direction_v2(no_empty_cand_viewidxs[query_cand], obs[i]['viewIndex'], spatial_answer=True))  # 可视化 GT answer
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

    def cal_dtw(self, shortest_distances, prediction, reference, success=None, threshold=3.0):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                best_previous_cost = min(
                    dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                cost = shortest_distances[prediction[i - 1]][reference[j - 1]]
                dtw_matrix[i][j] = cost + best_previous_cost

        dtw = dtw_matrix[len(prediction)][len(reference)]
        ndtw = np.exp(-dtw / (threshold * len(reference)))
        if success is None:
            success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
        sdtw = success * ndtw

        return {
            'DTW': dtw,
            'nDTW': ndtw,
            'SDTW': sdtw
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

    def prepare_self_improving_cot(self, obs, feedback, data_type, nav_vpids, nav_targets, traj, gmaps, t, cls_token, cot_summarization=None,land_pad_token=None):
        batch_navigation_cot_gt = []
        direction_of_gt_list = []
        ### get navigation cot GT
        for i, ob in enumerate(obs):
            if nav_targets[i] == -100:
                nav_target = 0
            else:
                nav_target = nav_targets[i]

            # common sense
            # if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug':
            if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug' \
                        or data_type[i] == 'cvdn' or data_type[i] == 'soon':
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

                    if self.args.cot_summarization:

                        direction_of_gt = self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']],
                                                                gmaps[i].node_positions[gt_vpid], obs[i]['heading'],
                                                                obs[i]['elevation'], spatial_relation=True)
                        direction_of_gt_list.append(direction_of_gt)
                        action_of_gt = ' '.join(self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']],
                                                                      gmaps[i].node_positions[gt_vpid],
                                                                      obs[i]['heading'], obs[i]['elevation']).split(
                            ' ')[:-1])
                        if self.args.landmark_not_merge_in_gt:
                            gt_landmarks = load_json(
                                os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid,
                                             f"{ob['scan']}_{gt_vpid}.json"))[str(viewidx)]
                            if len(gt_landmarks) > self.args.land_num:
                                gt_landmarks = random.sample(gt_landmarks, self.args.land_num)
                            # print(f"gt_landmarks: {gt_landmarks}")
                            if self.args.mlm:
                                common_sense = f"I should {direction_of_gt} to an observation with {', '.join(gt_landmarks)}."
                            else:
                                common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                        else:
                            gt_landmarks = cot_summarization[i][action_of_gt]['landmarks']
                            #common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                            if self.args.mlm:
                                common_sense = f"I should {direction_of_gt} to an observation with {', '.join(gt_landmarks)}."
                            else:
                                common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                    else:
                        common_sense = f"An observation with [{', '.join(gt_landmarks)}] may match with short-term goal and is likely to lead to the long-term goal."
                        #common_sense = f"I should {direction_of_gt} + ' to an observation with {', '.join(gt_landmarks)}."
                else:
                    direction_of_gt_list.append("stop")
                    if self.args.mlm:
                        common_sense = f"I should stop at to an observation."
                    else:
                        common_sense = f"Observation matches with long-term goal, so stop."


                if self.args.cot_v4 or self.args.cot_first_in_gt:
                    common_sense = f"{common_sense}"
                else:
                    common_sense = f"- Reasoning: {common_sense}"

            # action prediction
            # if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug':
            if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug' \
                        or data_type[i] == 'cvdn' or data_type[i] == 'soon':
                if self.args.action_first_in_gt:
                    action_prediction = '{}'.format(cls_token)
                else:
                    action_prediction = '- Action Decision: {}'.format(cls_token)

            if self.args.cot_summarization:
                if self.args.cot_v4:
                    navigation_cot_components = [common_sense]
                elif self.args.action_first_in_gt:
                    navigation_cot_components = [action_prediction, common_sense]
                elif self.args.cot_first_in_gt:
                    navigation_cot_components = [common_sense, action_prediction]
                else:
                    navigation_cot_components = [common_sense, action_prediction]
            # else:
            #     navigation_cot_components = [long_term_goal, short_term_goal, common_sense,
            #                                  action_prediction]  # TODO: spatial relation
            # navigation_cot_components = [long_term_goal, short_term_goal, common_sense] # TODO: spatial relation
            # navigation_cot_components = [cls_token]
            if self.args.cot_v4:
                navigation_cot_gt = navigation_cot_components[0]
            else:
                navigation_cot_gt = '\n'.join(navigation_cot_components)
            batch_navigation_cot_gt.append(navigation_cot_gt)
            # print(f"batch_navigation_cot_gt:{batch_navigation_cot_gt}")
        # if self.args.self_improving_cot:
        #     return batch_navigation_cot_gt, direction_of_gt_list
        # else:
        #     return batch_navigation_cot_gt
        return batch_navigation_cot_gt, direction_of_gt_list

    def prepare_cot(self, obs, feedback, data_type, nav_vpids, nav_targets, traj, gmaps, t, cls_token, cot_summarization=None,land_pad_token=None):
        batch_navigation_cot_gt = []
        direction_of_gt_list = []
        ### get navigation cot GT
        for i, ob in enumerate(obs):
            if nav_targets[i] == -100:
                nav_target = 0
            else:
                nav_target = nav_targets[i]

            # # long-term goal
            # if data_type[i] == 'r2r' or data_type[i] == 'r2r_aug':
            #     if 'fg_instruction' in ob:
            #         long_term_goal = ob['fg_instruction'][ob['fg_view'][-1]]
            #         if ob['fg_view'][-1] != len(ob['fg_instruction']) - 1:
            #             long_term_goal = ', '.join([long_term_goal, ob['fg_instruction'][-1]]) + '.'
            #         else:
            #             long_term_goal = ob['fg_instruction'][ob['fg_view'][-1]] + '.'
            #     else:
            #         long_term_goal = ob['instruction']
            # if data_type[i] == 'reverie':
            #     long_term_goal = ob['instruction']
            # long_term_goal = f"- Long-term Goal: {long_term_goal}"

            # short-term goal
            #if data_type[i] == 'r2r' or data_type[i] == 'r2r_aug':
            # if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug' \
            #         or data_type[i] == 'cvdn' or data_type[i] == 'soon':
            #     if feedback == 'teacher' and 'fg_instruction' in ob:
            #         if t >= len(ob['fg_instruction']):
            #             short_term_goal = ob['fg_instruction'][-1]
            #         else:
            #             short_term_goal = ob['fg_instruction'][t]
            #     if feedback == 'sample' or 'fg_instruction' not in ob or feedback == 'argmax':
            #         gt_vpid = nav_vpids[i][nav_target]
            #         if gt_vpid is not None:
            #             gt_sub_path = gmaps[i].graph.path(ob['viewpoint'], gt_vpid)
            #             if len(gt_sub_path) == 1:
            #                 prev_vp = traj[i]['path'][-1][-1]
            #             else:
            #                 prev_vp = gt_sub_path[-2]
            #             viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][gt_vpid]
            #             gt_landmarks = load_json(
            #                 os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid, f"{ob['scan']}_{gt_vpid}.json"))[str(viewidx)]
            #             short_term_goal = f"Go to the direction of [{', '.join(gt_landmarks)}]"
            #         else:
            #             short_term_goal = "Stop"
            # if data_type[i] == 'reverie':
            #     gt_vpid = nav_vpids[i][nav_target]
            #     if gt_vpid is not None:
            #         # traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], gt_vpid))
            #         gt_sub_path = gmaps[i].graph.path(ob['viewpoint'], gt_vpid)
            #         if len(gt_sub_path) == 1:
            #             prev_vp = traj[i]['path'][-1][-1]
            #         else:
            #             prev_vp = gt_sub_path[-2]
            #         viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][gt_vpid]
            #         gt_landmarks = load_json(
            #             os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid, f"{ob['scan']}_{gt_vpid}.json"))[str(viewidx)]
            #         short_term_goal = f"Go to the direction of [{', '.join(gt_landmarks)}]"
            #     else:
            #         short_term_goal = "Stop"
            # short_term_goal = f"- Short-term Goal: {short_term_goal}."

            # spatial relation
            # if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug':
            #     cand_viewidxs, cand_position = [], []
            #     cand_spatial_relation = []
            #     for x, vpid in enumerate(nav_vpids[i]):
            #         found_cc = False
            #         if x == 0:
            #             cand_spatial_relation.append(f"Cand ({x}) means to stop.")
            #         else:
            #             for j, cc in enumerate(ob['candidate']):
            #                 if cc['viewpointId'] == vpid:
            #                     cand_viewidxs.append(cc['pointId'])
            #                     cand_spatial_relation.append(f"Cand ({x}) shows [{', '.join(ob['landmarks'][str(cc['pointId'])])}] {self.get_direction_v2(cc['pointId'], obs[i]['viewIndex'], spatial_relation=True)} me.")
            #                     found_cc = True
            #             # if not found_cc:
            #             #     cand_spatial_relation.append(f"Cand ({x}) means to go back to the previous unexplored direction.")
            #     spatial_relation = ' '.join(cand_spatial_relation)
            # spatial_relation = f"- Spatial Relation: {spatial_relation}"

            # common sense
            #if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug':
            if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug' \
                        or data_type[i] == 'cvdn' or data_type[i] == 'soon':
                if self.args.random_target_vp_in_cot_gt:
                    navigable_wogt_vpids = [x for x in range(len(nav_vpids[i])) if cand_masks[i][x]]
                    random_vpid_idx = random.sample(navigable_wogt_vpids,1)[0]
                    gt_vpid = nav_vpids[i][random_vpid_idx]
                    print(f"random choosed gt vpid for cot gt: {random_vpid_idx},{gt_vpid}  real gt vpid: {nav_target},{nav_vpids[i][nav_target]}")
                else:
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

                    if self.args.cot_summarization:

                        direction_of_gt = self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']], gmaps[i].node_positions[gt_vpid], obs[i]['heading'], obs[i]['elevation'], spatial_relation=True)
                        direction_of_gt_list.append(direction_of_gt)
                        action_of_gt = ' '.join(self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']], gmaps[i].node_positions[gt_vpid], obs[i]['heading'], obs[i]['elevation']).split(' ')[:-1])
                        if self.args.landmark_not_merge_in_gt:
                            gt_landmarks = load_json(
                                os.path.join(self.t2t_landmark_dir, ob['scan'], gt_vpid,
                                             f"{ob['scan']}_{gt_vpid}.json"))[str(viewidx)]
                            if len(gt_landmarks)>self.args.land_num:
                                gt_landmarks = random.sample(gt_landmarks,self.args.land_num)
                            # print(f"gt_landmarks: {gt_landmarks}")
                        else:
                            gt_landmarks = cot_summarization[i][action_of_gt]['landmarks']
                        if self.args.cot_v4_only_direction:
                            common_sense = f"I should go to an observation {direction_of_gt} me."
                        elif self.args.cot_v4_only_landmark:
                            common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}]."
                        else:
                            if self.args.mlm:
                                common_sense = {}

                                common_sense["dir_1"] = direction_of_gt.split(" ")[0]
                                common_sense["dir_2"] = direction_of_gt.split(" ")[1]
                                # common_sense["land"] = []
                                # for landmark_ind in range(len(gt_landmarks)):
                                #     if " " in gt_landmarks[landmark_ind]:
                                #         common_sense["land"].append(gt_landmarks[landmark_ind].split(" ")[-1])
                                #     else:
                                #         common_sense["land"].append(gt_landmarks[landmark_ind])
                                # ori_common_sense_len = len(common_sense["land"])
                                # if ori_common_sense_len < 5:
                                #     for _ in range(5-ori_common_sense_len):
                                #         #common_sense["land"].append(land_pad_token)
                                #         common_sense["land"].append('room')
                                common_sense["land"] = ""
                                for landmark_ind in range(len(gt_landmarks)):
                                    common_sense["land"] += gt_landmarks[landmark_ind] + ", "
                                common_sense["land"] = common_sense["land"][:-2]
                                # if len(common_sense["land"]) < self.args.land_token_region_length:
                                #     common_sense["land"] += (self.args.land_token_region_length-len(common_sense["land"])) * land_pad_token
                                print('common_sense_land')
                                print(common_sense["land"])
                            else:
                                common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                    else:
                        common_sense = f"An observation with [{', '.join(gt_landmarks)}] may match with short-term goal and is likely to lead to the long-term goal."
                else:
                    direction_of_gt_list.append("stop")
                    if self.args.mlm:
                        common_sense = {}
                        common_sense["dir_1"] = "just"
                        common_sense["dir_2"] = "stop"
                        common_sense["land"] = ""
                        # common_sense["land"] += self.args.land_token_region_length * land_pad_token
                        # print('common_sense_land')
                        # print(common_sense["land"])
                        #common_sense["land"] = []
                        # for _ in range(5):
                        #     #common_sense["land"].append(land_pad_token)
                        #     common_sense["land"].append('room')
                    else:
                        print('gt_vpid is None!!!!!!')
                        common_sense = f"Observation matches with long-term goal, so stop."

                if self.args.cot_v4 or self.args.cot_first_in_gt:
                    if not self.args.mlm:
                        common_sense = f"{common_sense}"
                else:
                    common_sense = f"- Reasoning: {common_sense}"

            # action prediction
            #if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug':
            if data_type[i] == 'r2r' or data_type[i] == 'reverie' or data_type[i] == 'r2r_aug' \
                        or data_type[i] == 'cvdn' or data_type[i] == 'soon':
                if self.args.action_first_in_gt:
                    action_prediction = '{}'.format(cls_token)
                else:
                    action_prediction = '- Action Decision: {}'.format(cls_token)

            if self.args.cot_summarization:
                if self.args.cot_v4:
                    navigation_cot_components = [common_sense]
                elif self.args.action_first_in_gt:
                    navigation_cot_components = [action_prediction, common_sense]
                elif self.args.cot_first_in_gt:
                    navigation_cot_components = [common_sense, action_prediction]
                else:
                    navigation_cot_components = [common_sense, action_prediction]
            else:
                navigation_cot_components = [long_term_goal, short_term_goal, common_sense, action_prediction] # TODO: spatial relation
            # navigation_cot_components = [long_term_goal, short_term_goal, common_sense] # TODO: spatial relation
            # navigation_cot_components = [cls_token]
            if self.args.cot_v4:
                navigation_cot_gt = navigation_cot_components[0]
            else:
                navigation_cot_gt = '\n'.join(navigation_cot_components)
            batch_navigation_cot_gt.append(navigation_cot_gt)
            # print(f"batch_navigation_cot_gt:{batch_navigation_cot_gt}")
        # if self.args.self_improving_cot:
        #     return batch_navigation_cot_gt, direction_of_gt_list
        # else:
        #     return batch_navigation_cot_gt
        return batch_navigation_cot_gt, direction_of_gt_list

    def prepare_summarization(self, obs, feedback, data_type, nav_vpids, nav_targets, traj, gmaps, t, vp_landmarks, cand_masks):
        """
        ### Summarization: Turn right are Candidate (1) containing [landmark 1, landmark 2, ...]. Turn left are Candidate (2) containing [landmark 1, landmark 2, ...]. Go forward are... Go back are... Go up are... Go down are...
        """

        batch_summary = []
        for i, ob in enumerate(obs):
            cur_nav_vpids = [nav_vpids[i][x] for x in range(len(nav_vpids[i])) if cand_masks[i][x]]

            summary = {}
            for j, vp in enumerate(cur_nav_vpids):
                if j > 0 :
                    vp_direction =' '.join(self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']],
                                                      gmaps[i].node_positions[vp], obs[i]['heading'], obs[i]['elevation']).split(' ')[:2]) # delete 'to' in direction

                    # vp_direction =' '.join(self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']],
                    #                                   gmaps[i].node_positions[vp], obs[i]['heading'], obs[i]['elevation'], spatial_relation=True)) # delete 'to' in direction

                    if vp_direction not in summary:
                        summary[vp_direction] = {}
                        summary[vp_direction]['landmarks'] = set()
                        summary[vp_direction]['cand_index'] = []
                    summary[vp_direction]['landmarks'].update(vp_landmarks[i][vp])
                    summary[vp_direction]['cand_index'].append(j)

            directions = list(summary.keys())
            for direction in directions:
                if len(summary[direction]['landmarks']) > self.args.land_num:
                    summary[direction]['landmarks'] = random.sample(summary[direction]['landmarks'], self.args.land_num)

            batch_summary.append(summary)
        return batch_summary

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

    def get_action_from_logit_rand(self, rand_perms, a_t, nav_vpids, gmap_visited_masks):
        bs = len(nav_vpids)

        action_ids = np.zeros(bs, dtype=np.int64)
        for i in range(bs):
            rand_perm_list = rand_perms[i].numpy().tolist()

            if a_t[i] == 0:
                action_ids[i] = 0
            elif a_t[i] == -100:
                action_ids[i] = 0
            else:
                action_ids[i] = rand_perm_list.index(a_t[i] - 1 - (gmap_visited_masks[i]==True).sum()) + 1  # add stop dimension

        return action_ids

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
            epoch=0,
            **kwargs
    ):
        dataset_cfg = config.Pretrain if args.stage == 'pretrain' else config.Multi
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)

        #if args.enable_RL_A2C:
            #critic = kwargs['critic']

        if args.enable_RL_A2C:
            if args.alternate_IL_RL:
                if step % 2 == 0:
                    # IL
                    loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                    )
                else:
                    # RL
                    loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="sample", train_ml=None,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,
                        #critic=critic
                    )
            elif args.alternate_dagger_RL:
                if step % 2 == 0:
                    print('rollout dagger!!!')
                    # dagger
                    loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="sample", train_ml=loss_coef,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                    )
                else:
                    # RL
                    print('rollout RL!!!')
                    loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="sample", train_ml=None,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,
                        #critic=critic
                    )
            else:
                loss, _ = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="sample", train_ml=None,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,
                    #critic=critic
                )
        else:
            if args.stage == 'pretrain' or step % 2 == 0 or args.only_IL:
                #################### imitation learning ####################
                if self.args.enable_self_select:
                    loss, self_select_loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,epoch=epoch
                    )
                else:
                    loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,epoch=epoch
                    )
            else:
                if self.args.enable_RL_A2C:
                    #################### RL training ####################
                    loss, _ = self.rollout(
                        args, name, config.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="sample", train_ml=None,
                        entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,epoch=epoch
                        #critic=critic
                    )
                else:
                    #################### dagger training ####################
                    if self.args.enable_self_select:
                        loss,self_select_loss, _ = self.rollout(
                            args, name, config.Optim, batch,
                            model=model, criterion=criterion, dataset=dataset,
                            feedback="sample", train_ml=loss_coef,
                            entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,epoch=epoch
                        )
                    else:
                        loss, _ = self.rollout(
                            args, name, config.Optim, batch,
                            model=model, criterion=criterion, dataset=dataset,
                            feedback="sample", train_ml=loss_coef,
                            entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric,epoch=epoch
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
        if self.args.enable_self_select:
            return loss * args.gradient_accumulation_step, self_select_loss * args.gradient_accumulation_step
        else:
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
        pbar = tqdm(loader, disable=args.rank != 0, total=len(loader))
        if args.rank == 0:
            pbar.set_description(f"[Eval:{name}] inference")
            print(f"[Eval:{name}] start inference batches={len(loader)} samples={len(dataset)}")
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

        progress_every = max(1, int(getattr(args, "eval_progress_every", 10)))
        for i, batch in enumerate(pbar):
            if self.args.enable_self_select:
                ml_loss, self_select_loss, traj = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=None, dataset=dataset,
                    feedback="sample" if args.do_sample else "argmax", train_ml=None,
                    entropy_metric=entropy_metric, instr_pred_metric=None,
                    validate=True, trie=trie
                )
            else:
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

            if args.rank == 0 and ((i + 1) == 1 or (i + 1) == len(loader) or (i + 1) % progress_every == 0):
                pbar.set_postfix({
                    "batch": f"{i + 1}/{len(loader)}",
                    "preds": len(results),
                })
                print(f"[Eval:{name}] inference progress batch={i + 1}/{len(loader)} predictions={len(results)}")

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

    def get_pos_and_neg_cot(self, generated_sentences_navigation_cot, samples_logits, nav_targets):
        nav_target = nav_targets[0].cpu().item() ### TODO: only support batch_size == 1, need to code for larger batch
        sample_logits_of_target = samples_logits[:, nav_target]
        pos_id = torch.max(sample_logits_of_target, 1)
        neg_id = torch.min(sample_logits_of_target, 1)
        pos_cot = generated_sentences_navigation_cot[pos_id]
        neg_cot = generated_sentences_navigation_cot[neg_id]
        return pos_cot, neg_cot

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
            epoch=0,
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
        print('data_type')
        print(data_type)

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
                'navigable_gmap_vpids': {},
                'gt_node': {},
                'pred_node': {},
                'gt_vpid': {},
                'pred_vpid': {},
                'gt_action_viewpoint': {},
                'gt_viewidx': {},
                'gt_landmarks': {},
                'direction_of_gt': {},
                'details': {},
                'prompts': {},
                'cot_decision_consistency': {},
                'updated_cot_gt': {},
                'epoch': epoch,
            } for ob in obs]
        else:
            traj = [{
                'instr_id': ob['instr_id'],
                'path': [[ob['viewpoint']]],
                'details': {},
            } for ob in obs]

        if self.args.visualize_cot:
            visualize_file = open('output/' + 'cot_visualization.txt', 'a')
            visualize = traj

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
        if self.args.enable_self_select:
            self_select_loss = 0.

        # newly added
        # enable_RL_A2C = self.args.enable_RL_A2C and feedback == 'sample' and not validate and 'critic' in kwargs
        enable_RL_A2C = self.args.enable_RL_A2C and feedback == 'sample' and not validate and train_ml is None

        if enable_RL_A2C:
            #critic = kwargs["critic"]

            rewards = []
            hidden_states = []
            policy_log_probs = []
            rl_masks = []

            if self.args.add_efficiency_reward:
                step_length = [0 for _ in batch_size]

            # Init the reward shaping
            last_dist = np.zeros(batch_size, np.float32)
            last_ndtw = np.zeros(batch_size, np.float32)
            for i, ob in enumerate(obs):  # The init distance from the view point to the target
                last_dist[i] = ob['distance']
                path_act = sum(traj[i]['path'], [])
                last_ndtw[i] = self.cal_dtw(self.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

            total = 0

        ##newly added
        vp_landmarks = [{} for ob in obs]

        # self_refine_flag = False
        # self_select_flag = False
        if enable_RL_A2C:
            max_action_len = 10
            print('set RL max action len 10 for saving cost!!!')
        for t in range(max_action_len):

            # if isinstance(model, DDP):


            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                #model.module.critic = model.module.critic.to(model.module.model_type)

                # multi-gpu
                if ended.all() or t == max_action_len - 1:
                    flag = True
                    context = nullcontext
                    #if enable_RL_A2C:
                    #critic_context = nullcontext
                else:
                    context = model.no_sync
                    #if enable_RL_A2C:
                    #    critic_context = model.critic.no_sync
                    #else:
                    #    critic_context = nullcontext
            else:
                #model.module.critic = model.module.critic.to(model.model_type)
                # single-gpu
                if ended.all() or t == max_action_len - 1:
                    flag = True
                    context = nullcontext
                    #if enable_RL_A2C:
                    #critic_context = nullcontext
                else:
                    context = nullcontext
                    #if enable_RL_A2C:
                    #critic_context = nullcontext

            with context():
                #with critic_context():

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
                            vp_landmarks[i][cc['viewpointId']]=[remove_article(item) for item in ob['landmarks'][f"{cc['pointId']}"]]
                nav_inputs.update({
                    'vp_landmarks': vp_landmarks
                })

                in_progress = torch.tensor(ended).logical_not()
                if ended.all():
                    in_progress[0] = True

                nav_vpids = nav_inputs['gmap_vpids']
                imitation_learning = feedback == 'teacher'


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
                #if data_type[0] in ['r2r', 'soon', 'reverie', 'r2r_aug', 'reverie_aug']:
                #if data_type[0] in ['r2r', 'soon', 'reverie', 'cvdn', 'r2r_aug', 'reverie_aug']:
                enable_navigation_cot = (feedback == 'teacher' or feedback == 'sample' or feedback == 'argmax') and args.enable_navigation_cot
                if enable_navigation_cot:

                    if self.args.cot_summarization:
                        cand_masks = torch.clone(nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())

                        nav_inputs.update({
                        "cot_summarization": self.prepare_summarization(obs, feedback=feedback, data_type=data_type,
                                                              nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                                                              gmaps=gmaps, t=t, vp_landmarks=vp_landmarks,cand_masks=cand_masks
                                                              )})

                    # if self.args.self_improving_cot:
                    #     navigation_cot_gt, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                    #                                           nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                    #                                           gmaps=gmaps, t=t,
                    #                                           cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else model.lang_model.cls_token[0]
                    #                                           , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                    #     # navigation_cot_gt_, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                    #     #                                       nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                    #     #                                       gmaps=gmaps, t=t,
                    #     #                                       cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else model.lang_model.cls_token[0]
                    #     #                                       , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                    # else:
                    if self.args.random_target_vp_in_cot_gt:
                        cand_masks = torch.clone(
                            nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())
                        navigation_cot_gt, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,nav_vpids=nav_vpids,nav_targets=nav_targets,traj=traj,gmaps=gmaps, t=t,
                                                                           cls_token=model.module.lang_model.cls_token[0] if hasattr(model,'module') else
                                                                           model.lang_model.cls_token[0]
                                                                           , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None,
                                                                           cand_masks=cand_masks)
                    else:
                        if self.args.mlm:
                            navigation_cot_gt, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                                                                               nav_vpids=nav_vpids, nav_targets=nav_targets,
                                                                               traj=traj,
                                                                               gmaps=gmaps, t=t,
                                                                               cls_token=model.module.lang_model.cls_token[
                                                                                   0] if hasattr(model, 'module') else
                                                                               model.lang_model.cls_token[0]
                                                                               , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None, \
                                                                               land_pad_token=model.module.lang_model.landmark_token[1] if hasattr(model,'module') else model.lang_model.landmark_token[1])
                        else:
                            navigation_cot_gt, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                                                                               nav_vpids=nav_vpids, nav_targets=nav_targets,
                                                                               traj=traj,
                                                                               gmaps=gmaps, t=t,
                                                                               cls_token=model.module.lang_model.cls_token[
                                                                                   0] if hasattr(model, 'module') else
                                                                               model.lang_model.cls_token[0]
                                                                               , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None)


                    direction_of_gt = direct_of_gt
                    nav_inputs.update({
                        "navigation_cot_gt": navigation_cot_gt,
                    "direction_of_gt":direct_of_gt })

                    if self.args.mlm:
                        nav_inputs["prompts"] = self.prepare_prompts(
                            "navigation_once_forward_cot_navigation",
                            nav_inputs,
                            cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                            model.lang_model.cls_token[0],
                            land_token = model.module.lang_model.landmark_token if hasattr(model,'module') else model.lang_model.landmark_token,
                            dir_token = model.module.lang_model.direction_token if hasattr(model,
                                                                               'module') else model.lang_model.direction_token

                        )

                    else:
                        nav_inputs["prompts"] = self.prepare_prompts(
                            "navigation_once_forward_cot_navigation",
                            nav_inputs,
                            cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                            model.lang_model.cls_token[0]
                        )
                    print('cot gt')
                    print(nav_inputs["navigation_cot_gt"])
                    # if self.args.check_cot_input_gt:
                    #     print(f"inputs: {nav_inputs['prompts']} gt: {nav_inputs['navigation_cot_gt']}")

                    # if self.args.random_train_with_self_output_cot:
                    #     random_train_with_self_output = (t % 2 == 0)
                    #     assert self.args.train_with_self_output_cot == False, f"set both 'random_train_with_self_output_cot' and 'train_with_self_output' to True, random is disabled"
                    # else:
                    #     random_train_with_self_output = False

                    if self.args.test_with_cot_gt and validate:
                        batch_generated_sentences_navigation_cot = []
                    else:
                        if self.args.self_improve_wo_orisft:
                            with torch.no_grad():
                                # do once greedy cot
                                greedy_output = model('navigation_once_forward_cot_navigation', nav_inputs,
                                               training=False,
                                               multiple_sample_cot=False,
                                               **kwargs)
                        else:
                            if self.args.cot_output_as_supervision and not validate:
                                with torch.no_grad():
                                    # output_generate = model('navigation_once_forward_cot_navigation', nav_inputs,
                                    #            training=False, **kwargs)
                                    if self.args.cot_v4:
                                        self.args.cot_v4 = False
                                        output_generate = model('navigation_once_forward_cot_navigation', nav_inputs,
                                                            training=False,
                                                            **kwargs)
                                        self.args.cot_v4 = True
                                    else:
                                        output_generate = model('navigation_once_forward_cot_navigation', nav_inputs,
                                                            training=False,
                                                            **kwargs)

                                batch_generated_sentences_navigation_cot = output_generate[
                                        "generated_sentences_navigation_cot"]

                                nav_logits = output_generate['fuse_logits']
                                new_cot_gt = []
                                if train_ml is not None:
                                    _, a_t = nav_logits.max(1)
                                    for i in range(len(navigation_cot_gt)):
                                        if a_t[i] == nav_targets[i]:
                                            if self.args.replace_cot_gt_with_prob:
                                                if random.random() < self.args.replace_cot_gt_prob:
                                                    new_cot_gt_i = batch_generated_sentences_navigation_cot[i]
                                                    new_cot_gt_i = new_cot_gt_i.replace('<s>','')
                                                    new_cot_gt_i = new_cot_gt_i.replace('</s>','')
                                                    # avoid bad output causing out of memory
                                                    if int(len(new_cot_gt_i)) < int(2 * len(navigation_cot_gt[i])) and len(new_cot_gt_i) > 5:
                                                        print("use cot output as new supervision!!!")
                                                        new_cot_gt.append(new_cot_gt_i)
                                                    else:
                                                        new_cot_gt.append(navigation_cot_gt[i])
                                                else:
                                                    new_cot_gt.append(navigation_cot_gt[i])
                                            else:
                                                new_cot_gt_i = batch_generated_sentences_navigation_cot[i]
                                                new_cot_gt_i = new_cot_gt_i.replace('<s>', '')
                                                new_cot_gt_i = new_cot_gt_i.replace('</s>', '')
                                                # avoid bad output causing out of memory and empty content
                                                if int(len(new_cot_gt_i)) < int(2 * len(navigation_cot_gt[i])) and len(new_cot_gt_i) > 5:
                                                    print("use cot output as new supervision!!!")
                                                    new_cot_gt.append(new_cot_gt_i)
                                                else:
                                                    new_cot_gt.append(navigation_cot_gt[i])
                                        else:
                                            new_cot_gt.append(navigation_cot_gt[i])
                                print("ori cot gt")
                                print(navigation_cot_gt)
                                print("new cot gt")
                                print(new_cot_gt)
                                #if new_cot_gt != navigation_cot_gt:

                                for ob_ind in range(len(obs)):
                                    traj[ob_ind]['navigation_cot_gt'][str(t)] = navigation_cot_gt
                                    traj[ob_ind]['updated_cot_gt'][str(t)] = new_cot_gt

                                nav_inputs.update({
                                    "navigation_cot_gt": new_cot_gt})
                                if self.args.self_improving_cot:
                                    batch_generated_sentences_navigation_cot = output_generate[
                                        "generated_sentences_navigation_cot"]
                                    samples_logits = output_generate['fuse_logits']
                                    direction_landmark_dict = output_generate["direction_landmark_dict"]
                                    rand_perms = output_generate["rand_perms"]
                                    # print('navigational sample logits')
                                    # print(samples_logits)
                                    print('navigational reasoning output')
                                    print(batch_generated_sentences_navigation_cot)
                                    print('prompts for generating navigational reasoning in self improving phase')
                                    print(output_generate["prompts"])
                            output = model('navigation_once_forward_cot_navigation', nav_inputs, training=not validate, multiple_sample_cot=self.args.multiple_sample_cot and validate and not self.args.greedy_first_eval, **kwargs)
                            print('prompts for navigation loss')
                            print(output["prompts"])
                        if not validate:
                            if not self.args.self_improve_wo_orisft:
                                # lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                                # if self.args.cot_output_as_supervision:
                                #     with torch.no_grad():
                                #         # output_generate = model('navigation_once_forward_cot_navigation', nav_inputs,
                                #         #            training=False, **kwargs)
                                #         output_generate = model('navigation_once_forward_cot_navigation', nav_inputs,
                                #                                 training=False,
                                #                                 multiple_sample_cot=self.args.multiple_sample_cot,get_cot_output=True,
                                #                                 **kwargs)
                                #     batch_generated_sentences_navigation_cot = output_generate[
                                #         "generated_sentences_navigation_cot"]
                                #     nav_logits = output_generate['fuse_logits']
                                #     new_cot_gt = []
                                #     if train_ml is not None:
                                #         _, a_t = nav_logits.max(1)
                                #         for i in range(len(navigation_cot_gt)):
                                #             if a_t[i] == nav_targets[i]:
                                #                 new_cot_gt.append(batch_generated_sentences_navigation_cot[i])
                                #             else:
                                #                 new_cot_gt.append(navigation_cot_gt[i])
                                #     nav_inputs.update({'new_cot_gt':new_cot_gt})
                                #     lm_logits = output['logits']
                                #     batch = collections.defaultdict(lambda: None, nav_inputs)
                                #     all_text = []
                                #     for bn in range(batch_size):
                                #         prompt = batch["prompts"][bn]
                                #
                                #         label = batch['new_cot_gt'][bn] + f"{model.module.lang_model.tokenizer.eos_token}"
                                #         # if self.args.check_cot_input_gt:
                                #         #     print(f"\n{prompt}{label}")
                                #         all_text.append([prompt, label])
                                #
                                #     text_input = model.module.lang_model.tokenize(all_text).to(batch['vp_img_embeds'].device)
                                #     labels = text_input['input_ids'].clone()
                                #     labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
                                #     shift_logits = lm_logits[..., :-1, :].contiguous()
                                #     shift_labels = labels[..., 1:].contiguous()
                                #     # Flatten the tokens
                                #     loss_fct = nn.CrossEntropyLoss()
                                #     shift_logits = shift_logits.view(-1, model.module.lang_model.config.vocab_size)
                                #     shift_labels = shift_labels.view(-1)
                                #     # Enable model parallelism
                                #     shift_labels = shift_labels.to(shift_logits.device)
                                #     lm_loss = loss_fct(shift_logits, shift_labels)
                                #
                                #     lm_loss = lm_loss * args.cotsum_loss_coef / batch_size / args.gradient_accumulation_step
                                #
                                # else:
                                lm_loss = output["loss"] * args.cotsum_loss_coef / batch_size / args.gradient_accumulation_step
                                # if self.args.loss_cal_v2:
                                #     lm_loss.backward()
                                # # print(f"{t} cot ntp loss backward successfully")
                                #     instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                                #     ml_loss += lm_loss.detach()
                            if self.args.self_improving_cot and not self.args.cot_output_as_supervision:
                                if self.args.mlm:

                                    direction_landmark_dict = output["direction_landmark_dict"]
                                    rand_perms = output["rand_perms"]
                                    batch_generated_sentences_navigation_cot = None
                                else:
                                    #if self.args.self_improving_cot and not self.args.cot_output_as_supervision:
                                    # with torch.no_grad():
                                    #self_cot_output = model('navigation_cot', nav_inputs, training=False, **kwargs)
                                    #batch_generated_sentences_navigation_cot = self_cot_output["generated_sentences_navigation_cot"]
                                    #batch_generated_sentences_navigation_cot = ['None.']
                                    with torch.no_grad():
                                        # output_generate = model('navigation_once_forward_cot_navigation', nav_inputs,
                                        #            training=False, **kwargs)
                                        output_generate = model('navigation_once_forward_cot_navigation', nav_inputs, training=False, multiple_sample_cot=self.args.multiple_sample_cot, **kwargs)
                                    batch_generated_sentences_navigation_cot = output_generate["generated_sentences_navigation_cot"]
                                    samples_logits = output_generate['fuse_logits']
                                    direction_landmark_dict = output_generate["direction_landmark_dict"]
                                    rand_perms = output_generate["rand_perms"]
                                    # print('navigational sample logits')
                                    # print(samples_logits)
                                    print('navigational reasoning output')
                                    print(batch_generated_sentences_navigation_cot)
                                    print('prompts for generating navigational reasoning in self improving phase')
                                    print(output_generate["prompts"])

                            if enable_RL_A2C:
                                h_t = output["cls_hidden_state"]
                                hidden_states.append(h_t)

                        # else:
                            # batch_generated_sentences_navigation_cot = []
                            # for i in range(batch_size):
                            #     if output["generated_sentences_navigation_cot"] is not None:
                            #         if self.args.multiple_sample_cot:
                            #             traj[i]['generated_sentences_navigation_cot'][t] = output["generated_sentences_navigation_cot"]
                            #         else:
                            #             traj[i]['generated_sentences_navigation_cot'][t] = output["generated_sentences_navigation_cot"][i]
                            #
                            #     #traj[i]['navigation_cot_gt'][t] = nav_inputs['navigation_cot_gt'][i]
                            #     traj[i]['gmap_vpids'][t] = nav_vpids[i]
                            #     traj[i]['navigable_gmap_vpids'][t] = []
                            #     # print("nav_vpids[i]")
                            #     # print(nav_vpids[i])
                            #     # print("cand_masks[i]")
                            #     # print(cand_masks[i])
                            #     # assert len(nav_vpids[i]) == len(nav_vpids[i])
                            #     for cand_ind in range(len(nav_vpids[i])):
                            #         if cand_masks[i][cand_ind]:
                            #             traj[i]['navigable_gmap_vpids'][t].append(nav_vpids[i][cand_ind])
                            #     traj[i]['gt_node'][t] = nav_targets[i].item()
                            #     traj[i]['prompts'][t] = output["prompts"][i]
                            #     #print(output["prompts"][i])
                            # batch_generated_sentences_navigation_cot = output["generated_sentences_navigation_cot"]
                            # print(f"batch_generated_sentences_navigation_cot:{batch_generated_sentences_navigation_cot}")
                else:
                    nav_inputs["prompts"] = self.prepare_prompts(
                        "navigation",
                        nav_inputs,
                        cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                        model.lang_model.cls_token[0]
                    )
                    output = model('navigation', nav_inputs)

                if not self.args.self_improve_wo_orisft:
                    nav_logits = output['fuse_logits']

                if validate and self.args.multiple_sample_cot:
                    nav_target = nav_targets[0].cpu().item() ### TODO: only support batch_size == 1, need to code for larger batch
                    if nav_target == self.args.ignoreid: # ended
                        nav_target = 0
                    if self.args.greedy_first_eval:
                        _, a_t = nav_logits.max(1)
                        if a_t != nav_target:
                            print("Using temperature sample")
                            output = model('navigation_once_forward_cot_navigation', nav_inputs,
                                           training=not validate,
                                           multiple_sample_cot=self.args.multiple_sample_cot and validate,
                                           **kwargs)
                            nav_logits = output['fuse_logits']
                            sample_logits_of_target = nav_logits[:, nav_target]
                            pos_id = torch.max(sample_logits_of_target, 0)[1]

                            nav_logits = nav_logits[pos_id, :].unsqueeze(0)

                            # feedback = 'teacher'
                            if nav_target == 0:
                                feedback = 'teacher'
                                print("switch to teacher feedback at stop")
                            else:
                                feedback = 'argmax'
                        else:
                            print("Using greedy")
                    else:
                        sample_logits_of_target = nav_logits[:, nav_target]
                        pos_id = torch.max(sample_logits_of_target,0)[1]

                        nav_logits = nav_logits[pos_id, :].unsqueeze(0)

                        # feedback = 'teacher'
                        if nav_target == 0:
                            feedback = 'teacher'
                            print("switch to teacher feedback at stop")
                        else:
                            feedback = 'argmax'

                if not validate and self.args.multiple_sample_cot:
                    nav_target = nav_targets[0].cpu().item()  ### TODO: only support batch_size == 1, need to code for larger batch
                    if nav_target == self.args.ignoreid:  # ended
                        nav_target = 0

                    _, a_t = greedy_output['fuse_logits'].max(1)
                    if a_t != nav_target:
                        print("Self improving: Using temperature sample")
                        need_backward=True
                        sample_probs = torch.softmax(samples_logits, dim=-1)
                        print("sample_probs")
                        print(sample_probs)
                        sample_probs_of_target = sample_probs[:, nav_target]
                        pos_id = torch.max(sample_probs_of_target, 0)[1] ### index of the positive sample
                        pos_cot = batch_generated_sentences_navigation_cot[pos_id]
                        neg_cot = greedy_output['generated_sentences_navigation_cot'][0]
                        nav_logits = samples_logits[pos_id, :].unsqueeze(0)
                        fuse_embeds = output_generate['fuse_embeds'][pos_id, :].unsqueeze(0)
                    else:
                        print("Not self improving")
                        need_backward=False
                        nav_logits = greedy_output['fuse_logits']
                        pos_cot = ''
                        neg_cot = ''
                        fuse_embeds = greedy_output['fuse_embeds']

                if not self.args.self_improve_wo_orisft:

                    nav_probs = torch.softmax(nav_logits / args.temperature, 1)

                    # if not enable_RL_A2C:
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
                        if args.cot_summarization:
                            if self.args.cot_v4 or self.args.cot_first_in_gt or self.args.action_first_in_gt:


                                if self.args.add_lmloss_with_prob:
                                    step_wise_cal_lmloss_prob = random.random()
                                    if 'r2r' in data_type:
                                        cal_lmloss_prob = self.args.cal_lmloss_prob
                                    else:
                                        cal_lmloss_prob = self.args.cal_lmloss_prob_nor2r
                                    if step_wise_cal_lmloss_prob < cal_lmloss_prob:
                                        print("calculate lm loss!!!")
                                        if not flag:
                                            cnt_loss += lm_loss
                                        print(
                                            f"step{t}: lm_loss:{lm_loss.detach()}, cnt_loss: {cnt_loss.detach()}, ml_loss: {ml_loss}")
                                    else:
                                        #lm_loss.detach()
                                        print(
                                            f"step{t}: cnt_loss: {cnt_loss.detach()}, ml_loss: {ml_loss}")


                                else:
                                    if not flag:
                                        cnt_loss += lm_loss
                                    print(
                                        f"step{t}: lm_loss:{lm_loss.detach()}, cnt_loss: {cnt_loss.detach()}, ml_loss: {ml_loss}")

                            # if args.self_improving_cot:
                            #     cnt_loss += self_refine_lm_loss
                            # print(f"step{t}: cnt_loss: {cnt_loss.detach()}, ml_loss: {ml_loss}")
                        ml_loss += cnt_loss.detach()

                        # if self.args.self_improving_cot:
                        #     cnt_loss += self_refine_lm_loss

                        ########### Single-Step Backward ###########
                        if not validate:
                            cnt_loss.backward()
                        cnt_loss = 0.
                else:
                    nav_probs = torch.softmax(nav_logits / args.temperature, 1)

                if feedback == 'teacher':  # imitation learning
                    a_t = nav_targets  # teacher forcing
                elif feedback == 'sample':
                    c = torch.distributions.Categorical(nav_probs.float())
                    entropy_metric.accumulate(c.entropy().sum().item() / batch_size)  # For log
                    entropys.append(c.entropy())  # For optimization
                    a_t = c.sample().detach()
                    if enable_RL_A2C:
                        policy_log_probs.append(c.log_prob(a_t))
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
                    if not self.args.self_improve_wo_orisft and not self.args.multiple_sample_cot:
                        fuse_embeds = output['fuse_embeds']

                    hist_vis[idx].append(fuse_embeds[idx][a_t[idx]])

                    if args.add_cand_landmark and len(hist_vis[idx])>=6:
                        hist_vis[idx] = hist_vis[idx][len(hist_vis[idx])-6:]
                        history[idx] = history[idx][len(history[idx])-6:]

                for i in range(batch_size):
                    if a_t[i] == -100:
                        continue
                    if enable_navigation_cot and validate and not ended[i]:
                        step_key = str(t)
                        cand_masks = torch.clone(nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())
                        pred_action = a_t[i].item() if hasattr(a_t[i], 'item') else int(a_t[i])
                        gt_action = nav_targets[i].item() if hasattr(nav_targets[i], 'item') else int(nav_targets[i])
                        if gt_action == self.args.ignoreid:
                            gt_action = 0

                        if output.get("generated_sentences_navigation_cot") is not None:
                            if self.args.multiple_sample_cot:
                                traj[i]['generated_sentences_navigation_cot'][step_key] = output["generated_sentences_navigation_cot"]
                            else:
                                traj[i]['generated_sentences_navigation_cot'][step_key] = output["generated_sentences_navigation_cot"][i]

                        traj[i]['navigation_cot_gt'][step_key] = navigation_cot_gt[i]
                        traj[i]['direction_of_gt'][step_key] = direction_of_gt[i] if i < len(direction_of_gt) else ''
                        traj[i]['gmap_vpids'][step_key] = nav_vpids[i]
                        traj[i]['navigable_gmap_vpids'][step_key] = []
                        for cand_ind in range(len(nav_vpids[i])):
                            if cand_masks[i][cand_ind]:
                                traj[i]['navigable_gmap_vpids'][step_key].append(nav_vpids[i][cand_ind])

                        traj[i]['gt_node'][step_key] = gt_action
                        traj[i]['pred_node'][step_key] = pred_action
                        traj[i]['gt_vpid'][step_key] = nav_vpids[i][gt_action] if gt_action < len(nav_vpids[i]) else None
                        traj[i]['pred_vpid'][step_key] = nav_vpids[i][pred_action] if pred_action < len(nav_vpids[i]) else None
                        traj[i]['gt_action_viewpoint'][step_key] = obs[i]['viewpoint']
                        traj[i]['gt_viewidx'][step_key] = None
                        if traj[i]['gt_vpid'][step_key] is not None:
                            scanvp = '%s_%s' % (obs[i]['scan'], obs[i]['viewpoint'])
                            traj[i]['gt_viewidx'][step_key] = self.scanvp_cands.get(scanvp, {}).get(traj[i]['gt_vpid'][step_key])
                        traj[i]['gt_landmarks'][step_key] = []
                        if traj[i]['gt_vpid'][step_key] is not None and traj[i]['gt_vpid'][step_key] in vp_landmarks[i]:
                            traj[i]['gt_landmarks'][step_key] = vp_landmarks[i][traj[i]['gt_vpid'][step_key]]
                        traj[i]['prompts'][step_key] = output["prompts"][i] if "prompts" in output else nav_inputs["prompts"][i]

                        if self.args.enable_action_reasoning_f1 and len(traj[i]['generated_sentences_navigation_cot']) <= self.args.action_reasoning_print_examples:
                            print("[ActionReasoningF1] step intermediate")
                            print({
                                "instr_id": traj[i]['instr_id'],
                                "step": step_key,
                                "pred_node": pred_action,
                                "gt_node": gt_action,
                                "pred_cot": traj[i]['generated_sentences_navigation_cot'].get(step_key, ''),
                                "gt_direction": traj[i]['direction_of_gt'].get(step_key, ''),
                                "gt_landmarks": traj[i]['gt_landmarks'].get(step_key, []),
                            })

                if not validate:
                    # if feedback == 'teacher' or feedback == 'sample':  # in training
                    assert feedback in ['teacher',
                                        'sample'], "Feedback must be either `teacher' or `sample' in training. "
                    a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
                else:
                    a_t_stop = a_t == 0



                # ########### Self-Refine Sub-task ###########
                enable_self_refine = args.self_improving_cot and args.enable_navigation_cot and not validate and args.enable_self_refine# TODO: check if need to add more conditions
                if enable_self_refine:
                    pano_inputs = self.panorama_feature_variable_object(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']

                    # navigation policy
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )

                    nav_inputs.update({
                        'vp_landmarks': vp_landmarks
                    })

                    nav_inputs.update({
                        'view_lens': pano_inputs['view_lens'],
                        'instruction': instructions,
                        'history': history,
                        'hist_vis': hist_vis,
                        'data_type': data_type
                    })

                    if self.args.cot_summarization:
                        cand_masks = torch.clone(nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())

                        nav_inputs.update({
                        "cot_summarization": self.prepare_summarization(obs, feedback=feedback, data_type=data_type,
                                                              nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                                                              gmaps=gmaps, t=t, vp_landmarks=vp_landmarks,cand_masks=cand_masks
                                                              )})
                    if self.args.mlm:
                        navigation_cot_gt, direction_of_gt = self.prepare_self_improving_cot(obs, feedback=feedback, data_type=data_type,
                                                              nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                                                              gmaps=gmaps, t=t,
                                                              cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else model.lang_model.cls_token[0]
                                                              , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                        nav_inputs.update({
                            "navigation_cot_gt": navigation_cot_gt})
                    elif self.args.cot_output_as_supervision:
                        nav_inputs.update({
                            "navigation_cot_gt": new_cot_gt})
                    else:
                        nav_inputs.update({
                            "navigation_cot_gt": navigation_cot_gt})
                    # if self.args.self_improving_cot:
                    #     navigation_cot_gt, direction_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                    #                      nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                    #                      gmaps=gmaps, t=t,
                    #                      cls_token=model.module.lang_model.cls_token[0] if hasattr(
                    #                          model, 'module') else model.lang_model.cls_token[0]
                    #                      , cot_summarization=nav_inputs[
                    #             'cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                    # else:
                    #     navigation_cot_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                    #                      nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                    #                      gmaps=gmaps, t=t,
                    #                      cls_token=model.module.lang_model.cls_token[0] if hasattr(
                    #                          model, 'module') else model.lang_model.cls_token[0]
                    #                      , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                    # nav_inputs.update({
                    #     "navigation_cot_gt": navigation_cot_gt})

                    if not self.args.multiple_sample_cot:
                        pos_cot = nav_inputs['navigation_cot_gt']
                        #---get neg---#
                        if self.args.landmark_not_merge_in_gt:
                            neg_cot = self.get_neg_cot(nav_inputs['navigation_cot_gt'], batch_generated_sentences_navigation_cot, direction_landmark_dict, direction_of_gt, data_type=data_type, nav_vpids=nav_vpids, nav_targets=nav_targets, cand_masks=cand_masks, gmaps=gmaps, obs=obs, traj=traj)
                        else:
                            neg_cot = self.get_neg_cot(nav_inputs['navigation_cot_gt'], batch_generated_sentences_navigation_cot, direction_landmark_dict, direction_of_gt)

                    nav_inputs.update({
                        "navigation_self_refine_pos": pos_cot,
                        "navigation_self_refine_neg": neg_cot
                    })

                    nav_inputs["prompts"] = self.prepare_prompts(
                        "navigation_self_refine",
                        nav_inputs
                    )

                    print("self refine gt")
                    print(nav_inputs["navigation_cot_gt"])
                    self_refine_output = model('navigation_cot', nav_inputs, training=not validate,rand_perms=rand_perms,
                                   **kwargs)
                    print("self refine prompts")
                    print(self_refine_output["prompts"])
                    if not validate:
                        # self_refine_lm_loss = self_refine_output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                        self_refine_lm_loss = self_refine_output["loss"] * args.self_refine_loss_weight / batch_size / args.gradient_accumulation_step

                        # if not need_backward:
                        #     with torch.no_grad():
                        #         self_refine_lm_loss = torch.tensor(0., dtype=self_refine_lm_loss.dtype).cuda(self_refine_lm_loss.device)
                        if self.args.add_selfrefine_loss_with_prob:
                            if 'r2r' in data_type:
                                cal_lmloss_prob = self.args.cal_lmloss_prob
                            else:
                                cal_lmloss_prob = self.args.cal_lmloss_prob_nor2r
                            if self.args.add_lmloss_with_prob:
                                if step_wise_cal_lmloss_prob < cal_lmloss_prob:
                                    self_refine_lm_loss.backward()
                                    # print(f"{t} cot ntp loss backward successfully")
                                    instr_pred_metric.accumulate(
                                        self_refine_lm_loss.detach().item() * args.gradient_accumulation_step)
                                    ml_loss += self_refine_lm_loss.detach()

                                    print(f"step{t}:　self_refine_lm_loss: {self_refine_lm_loss.detach()}")
                            else:
                                if random.random() < cal_lmloss_prob:
                                    self_refine_lm_loss.backward()
                                    # print(f"{t} cot ntp loss backward successfully")
                                    instr_pred_metric.accumulate(
                                        self_refine_lm_loss.detach().item() * args.gradient_accumulation_step)
                                    ml_loss += self_refine_lm_loss.detach()

                                    print(f"step{t}:　self_refine_lm_loss: {self_refine_lm_loss.detach()}")

                        else:
                            self_refine_lm_loss.backward()
                            # print(f"{t} cot ntp loss backward successfully")
                            instr_pred_metric.accumulate(self_refine_lm_loss.detach().item() * args.gradient_accumulation_step)
                            ml_loss += self_refine_lm_loss.detach()

                            print(f"step{t}:　self_refine_lm_loss: {self_refine_lm_loss.detach()}")

                    # else:
                    #     # for i in range(batch_size):
                    #         # traj[i]['generated_sentences_navigation_cot'][t] = \
                    #         # output["generated_sentences_navigation_cot"][i]
                    #         # traj[i]['navigation_cot_gt'][t] = nav_inputs['navigation_cot_gt'][i]
                    #         # traj[i]['gmap_vpids'][t] = nav_vpids[i]
                    #         # traj[i]['gt_node'][t] = nav_targets[i].item()
                    #     batch_generated_sentences_navigation_cot_navigation_self_refine = self_refine_output["generated_sentences_navigation_cot"]
                    #     print(f"batch_generated_sentences_navigation_cot_navigation_self_refine:{batch_generated_sentences_navigation_cot_navigation_self_refine}")


                ########### Self-Select Sub-task ###########
                enable_self_select = args.self_improving_cot and args.enable_navigation_cot and not validate and args.enable_self_select# TODO: check if need to add more conditions
                if enable_self_select:
                    pano_inputs = self.panorama_feature_variable_object(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']

                    # navigation policy
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )
                    nav_inputs.update({
                        'vp_landmarks': vp_landmarks
                    })

                    nav_inputs.update({
                        'view_lens': pano_inputs['view_lens'],
                        'instruction': instructions,
                        'history': history,
                        'hist_vis': hist_vis,
                        'data_type': data_type
                    })

                    if self.args.cot_summarization:
                        cand_masks = torch.clone(nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())

                        nav_inputs.update({
                        "cot_summarization": self.prepare_summarization(obs, feedback=feedback, data_type=data_type,
                                                              nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                                                              gmaps=gmaps, t=t, vp_landmarks=vp_landmarks,cand_masks=cand_masks
                                                              )})
                    if self.args.mlm:
                        navigation_cot_gt, direction_of_gt = self.prepare_self_improving_cot(obs, feedback=feedback, data_type=data_type,
                                                              nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                                                              gmaps=gmaps, t=t,
                                                              cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else model.lang_model.cls_token[0]
                                                              , cot_summarization=nav_inputs['cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                        nav_inputs.update({
                            "navigation_cot_gt": navigation_cot_gt})
                    elif self.args.cot_output_as_supervision:
                        nav_inputs.update({
                            "navigation_cot_gt": new_cot_gt})
                    else:
                        nav_inputs.update({
                            "navigation_cot_gt": navigation_cot_gt})
                    # if self.args.self_improving_cot:
                    #     navigation_cot_gt, direction_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                    #                      nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                    #                      gmaps=gmaps, t=t,
                    #                      cls_token=model.module.lang_model.cls_token[0] if hasattr(
                    #                          model, 'module') else model.lang_model.cls_token[0]
                    #                      , cot_summarization=nav_inputs[
                    #             'cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                    # else:
                    #     navigation_cot_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                    #                      nav_vpids=nav_vpids, nav_targets=nav_targets, traj=traj,
                    #                      gmaps=gmaps, t=t,
                    #                      cls_token=model.module.lang_model.cls_token[0] if hasattr(
                    #                          model, 'module') else model.lang_model.cls_token[0]
                    #                      , cot_summarization=nav_inputs[
                    #             'cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                    # nav_inputs.update({
                    #     "navigation_cot_gt": navigation_cot_gt})
                    if not self.args.multiple_sample_cot:
                        pos_cot = nav_inputs['navigation_cot_gt']
                        if self.args.landmark_not_merge_in_gt:
                            neg_cot = self.get_neg_cot(nav_inputs['navigation_cot_gt'], batch_generated_sentences_navigation_cot, direction_landmark_dict, direction_of_gt,data_type=data_type, nav_vpids=nav_vpids, nav_targets=nav_targets, cand_masks=cand_masks, gmaps=gmaps, obs=obs, traj=traj)
                        else:
                            #---get neg---#
                            neg_cot = self.get_neg_cot(nav_inputs['navigation_cot_gt'], batch_generated_sentences_navigation_cot, direction_landmark_dict, direction_of_gt)


                    nav_inputs.update({
                        "navigation_self_select_pos": pos_cot,
                        "navigation_self_select_neg": neg_cot
                    })

                    batch_cot_pair = []
                    batch_self_select_gt = []
                    for i in range(batch_size):
                        order = list(range(2)) #0,1
                        random.shuffle(order) #1,0
                        ori_pair = [nav_inputs['navigation_self_select_pos'][i], nav_inputs["navigation_self_select_neg"][i]]
                        new_pair = [ori_pair[x] for x in order]
                        self_select_gt = f'Output {order.index(0)+1}.' # pos_cot is at index 0 in ori_pair
                        batch_cot_pair.append(new_pair)
                        batch_self_select_gt.append(self_select_gt)
                    nav_inputs.update({
                        'navigation_cot_pair': batch_cot_pair,
                        "navigation_cot_gt": batch_self_select_gt
                    })

                    nav_inputs["prompts"] = self.prepare_prompts(
                        "navigation_self_select",
                        nav_inputs
                    )

                    # print("self select gt")
                    # print(nav_inputs["navigation_cot_gt"])
                    self_select_output = model('navigation_cot', nav_inputs, training=not validate, rand_perms=rand_perms,
                                   **kwargs)
                    print("self select prompts & gt")
                    print(self_select_output["prompts"], nav_inputs["navigation_cot_gt"])
                    if not validate:
                        # self_select_lm_loss = self_select_output["loss"] * args.gen_loss_coef * args.self_select_loss_weight / batch_size / args.gradient_accumulation_step
                        self_select_lm_loss = self_select_output["loss"] * args.self_select_loss_weight / batch_size / args.gradient_accumulation_step
                        self_select_loss += self_select_lm_loss.detach()
                        # if self.args.add_lmloss_with_prob:
                        #     if random.random() < self.args.cal_lmloss_prob:
                        #         print("calculate self select loss!!!")
                        #         print(f"step{t}: self_select_loss:{self_select_lm_loss.detach()}")
                        #     else:
                        #         self_select_lm_loss = torch.tensor(0.0, dtype=self_select_lm_loss.dtype,device=self_select_lm_loss.device, requires_grad=True)
                        # else:
                        if self.args.add_selfselect_loss_with_prob:
                            if 'r2r' in data_type:
                                cal_lmloss_prob = self.args.cal_lmloss_prob
                            else:
                                cal_lmloss_prob = self.args.cal_lmloss_prob_nor2r
                            if self.args.add_lmloss_with_prob:
                                if step_wise_cal_lmloss_prob < cal_lmloss_prob:
                                    self_select_lm_loss.backward()
                                    # print(f"{t} cot ntp loss backward successfully")
                                    instr_pred_metric.accumulate(
                                        self_select_lm_loss.detach().item() * args.gradient_accumulation_step)
                                    ml_loss += self_select_lm_loss.detach()
                                    print(f"step{t}: self_select_loss:{self_select_lm_loss.detach()}")
                            else:
                                if random.random() < cal_lmloss_prob:
                                    self_select_lm_loss.backward()
                                    # print(f"{t} cot ntp loss backward successfully")
                                    instr_pred_metric.accumulate(
                                        self_select_lm_loss.detach().item() * args.gradient_accumulation_step)
                                    ml_loss += self_select_lm_loss.detach()
                                    print(f"step{t}: self_select_loss:{self_select_lm_loss.detach()}")
                        else:
                            self_select_lm_loss.backward()
                            # print(f"{t} cot ntp loss backward successfully")
                            instr_pred_metric.accumulate(self_select_lm_loss.detach().item() * args.gradient_accumulation_step)
                            ml_loss += self_select_lm_loss.detach()
                            print(f"step{t}: self_select_loss:{self_select_lm_loss.detach()}")
                        # print(f"step{t}:　self_select_lm_loss: {self_select_lm_loss.detach()} self_refine_lm_loss: {self_refine_lm_loss.detach()}")

                    # else:
                    #     # for i in range(batch_size):
                    #         # traj[i]['generated_sentences_navigation_cot'][t] = \
                    #         # output["generated_sentences_navigation_cot"][i]
                    #         # traj[i]['navigation_cot_gt'][t] = nav_inputs['navigation_cot_gt'][i]
                    #         # traj[i]['gmap_vpids'][t] = nav_vpids[i]
                    #         # traj[i]['gt_node'][t] = nav_targets[i].item()
                    #     batch_generated_sentences_navigation_cot_navigation_self_select = self_select_output["generated_sentences_navigation_cot"]
                    #     print(f"batch_generated_sentences_navigation_cot_navigation_self_select:{batch_generated_sentences_navigation_cot_navigation_self_select}")


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
                x = 0
                for i, ob in enumerate(obs):
                    if 'fg_instruction' in ob and 'fg_view' in ob:
                        x+=1
                fgr2r_flag = x == len(obs)
                enable_fgr2r = (feedback == 'teacher') and (not flag) and (not a_t_stop[0]) and (data_type[0] == 'r2r') and (not validate) and fgr2r_flag and args.enable_fgr2r
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
                        print(f"step{t}: fgr2r_loss:{lm_loss.detach()}")

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
                        # print(f"step{t}: sum_loss:{lm_loss.detach()}")
                    else:
                        for i in range(batch_size):
                            generated_sentences = output["generated_sentences"]
                            traj[i]['generated_sentences'] = generated_sentences[i]
                            traj[i]['answer'] = nav_inputs['answer'][i]


                ########### Navigation RL Last-value ###########
                if enable_RL_A2C and flag:
                    if enable_navigation_cot:
                        pano_inputs = self.panorama_feature_variable_object(obs)
                        panorama_out = model('panorama', pano_inputs)
                        pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out[
                            'pano_masks']

                        # navigation policy
                        nav_inputs = self.nav_gmap_variable(obs, gmaps)
                        nav_inputs.update(
                            self.nav_vp_variable(
                                obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                                pano_inputs['view_lens'], pano_inputs['nav_types'],
                            )
                        )

                        nav_inputs.update({
                            'vp_landmarks': vp_landmarks
                        })

                        nav_inputs.update({
                            'view_lens': pano_inputs['view_lens'],
                            'instruction': instructions,
                            'history': history,
                            'hist_vis': hist_vis,
                            'data_type': data_type
                        })

                        if self.args.cot_summarization:
                            cand_masks = torch.clone(
                                nav_inputs['gmap_masks'] & nav_inputs[
                                    'gmap_visited_masks'].logical_not())

                            nav_inputs.update({
                                "cot_summarization": self.prepare_summarization(obs, feedback=feedback,
                                                                                data_type=data_type,
                                                                                nav_vpids=nav_vpids,
                                                                                nav_targets=nav_targets,
                                                                                traj=traj,
                                                                                gmaps=gmaps, t=t,
                                                                                vp_landmarks=vp_landmarks,
                                                                                cand_masks=cand_masks
                                                                                )})


                        nav_inputs["prompts"] = self.prepare_prompts(
                            "navigation_once_forward_cot_navigation",
                            nav_inputs,
                            cls_token=model.module.lang_model.cls_token[0] if hasattr(model,
                                                                                      'module') else
                            model.lang_model.cls_token[0]
                        )
                        # print('cot gt')
                        # print(nav_inputs["navigation_cot_gt"])
                        # if args.enable_generative_action_prediction or args.add_action_prediction_in_navigational_reasoning:
                        #     nav_inputs["nav_targets"] = nav_targets
                        # if self.args.check_cot_input_gt:
                        #     print(f"inputs: {nav_inputs['prompts']} gt: {nav_inputs['navigation_cot_gt']}")

                        # if self.args.random_train_with_self_output_cot:
                        #     random_train_with_self_output = (t % 2 == 0)
                        #     assert self.args.train_with_self_output_cot == False, f"set both 'random_train_with_self_output_cot' and 'train_with_self_output' to True, random is disabled"
                        # else:
                        #     random_train_with_self_output = False



                        output = model('navigation_once_forward_cot_navigation', nav_inputs,
                                       training=False,
                                       **kwargs)

                        # last_value__ = model.critic(output['cls_hidden_state'].to(torch.float32)).detach()  # The value esti of the last state, remove the grad for safety
                        #last_value__ = model.module.critic(output['cls_hidden_state']).detach()  # The value esti of the last state, remove the grad for safety
                        #nav_inputs.update({'state':output['cls_hidden_state']})
                        last_value__ = model(mode='critic', batch=nav_inputs,state=output['cls_hidden_state']).detach()  # The value esti of the last state, remove the grad for safety

                        discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
                        for i in range(batch_size):
                            if not ended[i]:  # If the action is not ended, use the value function as the last reward
                                discount_reward[i] = last_value__[i]

                # ########### Navigation RL Last-value ###########
                # if enable_RL_A2C and flag:
                #     if enable_navigation_cot:
                #         pano_inputs = self.panorama_feature_variable_object(obs)
                #         panorama_out = model('panorama', pano_inputs)
                #         pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                #
                #         # navigation policy
                #         nav_inputs = self.nav_gmap_variable(obs, gmaps)
                #         nav_inputs.update(
                #             self.nav_vp_variable(
                #                 obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                #                 pano_inputs['view_lens'], pano_inputs['nav_types'],
                #             )
                #         )
                #
                #         nav_inputs.update({
                #             'vp_landmarks': vp_landmarks
                #         })
                #
                #         nav_inputs.update({
                #             'view_lens': pano_inputs['view_lens'],
                #             'instruction': instructions,
                #             'history': history,
                #             'hist_vis': hist_vis,
                #             'data_type': data_type
                #         })
                #
                #         if self.args.cot_summarization:
                #             cand_masks = torch.clone(
                #                 nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())
                #
                #             nav_inputs.update({
                #                 "cot_summarization": self.prepare_summarization(obs, feedback=feedback,
                #                                                                 data_type=data_type,
                #                                                                 nav_vpids=nav_vpids,
                #                                                                 nav_targets=nav_targets,
                #                                                                 traj=traj,
                #                                                                 gmaps=gmaps, t=t,
                #                                                                 vp_landmarks=vp_landmarks,
                #                                                                 cand_masks=cand_masks
                #                                                                 )})
                #
                #         # if self.args.self_improving_cot:
                #         #     navigation_cot_gt, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                #         #                                                        nav_vpids=nav_vpids, nav_targets=nav_targets,
                #         #                                                        traj=traj,
                #         #                                                        gmaps=gmaps, t=t,
                #         #                                                        cls_token=model.module.lang_model.cls_token[
                #         #                                                            0] if hasattr(model, 'module') else
                #         #                                                        model.lang_model.cls_token[0]
                #         #                                                        , cot_summarization=nav_inputs[
                #         #             'cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                #         # else:
                #         #     navigation_cot_gt, direct_of_gt = self.prepare_cot(obs, feedback=feedback, data_type=data_type,
                #         #                                                        nav_vpids=nav_vpids, nav_targets=nav_targets,
                #         #                                                        traj=traj,
                #         #                                                        gmaps=gmaps, t=t,
                #         #                                                        cls_token=model.module.lang_model.cls_token[
                #         #                                                            0] if hasattr(model, 'module') else
                #         #                                                        model.lang_model.cls_token[0]
                #         #                                                        , cot_summarization=nav_inputs[
                #         #             'cot_summarization'] if 'cot_summarization' in nav_inputs else None)
                #         # nav_inputs.update({
                #         #     "navigation_cot_gt": navigation_cot_gt,
                #         #     "direction_of_gt": direct_of_gt})
                #         if self.args.cot_v5:
                #             nav_inputs["prompts"] = self.prepare_prompts(
                #                 "navigation_once_forward_cot_navigation",
                #                 nav_inputs,
                #                 cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                #                 model.lang_model.cls_token[0],
                #                 land_token=model.module.lang_model.cls_token[2] if hasattr(model, 'module') else
                #                 model.lang_model.cls_token[2],
                #                 dir_token=model.module.lang_model.cls_token[3] if hasattr(model, 'module') else
                #                 model.lang_model.cls_token[3],
                #             )
                #             nav_inputs["land_tokens"] = model.module.lang_model.landmark_token if hasattr(model,
                #                                                                                           'module') else model.lang_model.landmark_token
                #             nav_inputs["dir_tokens"] = model.module.lang_model.direction_token if hasattr(model,
                #                                                                                           'module') else model.lang_model.direction_token
                #         else:
                #             nav_inputs["prompts"] = self.prepare_prompts(
                #                 "navigation_once_forward_cot_navigation",
                #                 nav_inputs,
                #                 cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                #                 model.lang_model.cls_token[0]
                #             )
                #         # print('cot gt')
                #         # print(nav_inputs["navigation_cot_gt"])
                #         # if args.enable_generative_action_prediction or args.add_action_prediction_in_navigational_reasoning:
                #         #     nav_inputs["nav_targets"] = nav_targets
                #         # if self.args.check_cot_input_gt:
                #         #     print(f"inputs: {nav_inputs['prompts']} gt: {nav_inputs['navigation_cot_gt']}")
                #
                #         # if self.args.random_train_with_self_output_cot:
                #         #     random_train_with_self_output = (t % 2 == 0)
                #         #     assert self.args.train_with_self_output_cot == False, f"set both 'random_train_with_self_output_cot' and 'train_with_self_output' to True, random is disabled"
                #         # else:
                #         #     random_train_with_self_output = False
                #
                #         if self.args.random_slow_and_fast:
                #             if random.random() < self.args.random_slow_and_fast_prob:
                #                 activate_fast = True
                #             else:
                #                 activate_fast = False
                #             nav_inputs["activate_fast"] = activate_fast
                #
                #         output = model('navigation_once_forward_cot_navigation', nav_inputs, training=False,
                #                        **kwargs)
                #
                #         last_value__ = critic(
                #             output['cls_hidden_state']).detach()  # The value esti of the last state, remove the grad for safety
                #
                #
                #         discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
                #         for i in range(batch_size):
                #             if not ended[i]:  # If the action is not ended, use the value function as the last reward
                #                 discount_reward[i] = last_value__[i]

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

                if enable_RL_A2C:

                    # Calculate the mask and reward
                    dist = np.zeros(batch_size, np.float32)
                    ndtw_score = np.zeros(batch_size, np.float32)
                    reward = np.zeros(batch_size, np.float32)
                    rl_mask = np.ones(batch_size, np.float32)
                    for i, ob in enumerate(obs):
                        dist[i] = ob['distance']
                        path_act = sum(traj[i]['path'], [])
                        ndtw_score[i] = \
                        self.cal_dtw(self.shortest_distances[ob['scan']], path_act, ob['gt_path'],
                                     threshold=3.0)['nDTW']
                        if ended[i]:
                            reward[i] = 0.0
                            rl_mask[i] = 0.0
                        else:
                            if self.args.add_efficiency_reward:
                                step_length[i] += 1
                            action_idx = cpu_a_t[i]
                            # Target reward
                            if action_idx == None:  # If the action now is end
                                if self.args.add_efficiency_reward:
                                    if dist[i] < 3.0:  # Correct
                                        efficiency_weight = 0.1
                                        reward[i] = 2.0 + ndtw_score[i] * 2.0 + step_length[i] * efficiency_weight
                                    else:  # Incorrect
                                        reward[i] = -2.0
                                else:
                                    if dist[i] < 3.0:  # Correct
                                        reward[i] = 2.0 + ndtw_score[i] * 2.0
                                    else:  # Incorrect
                                        reward[i] = -2.0
                            else:  # The action is not end
                                # Path fidelity rewards (distance & nDTW)
                                reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                                ndtw_reward = ndtw_score[i] - last_ndtw[i]
                                if reward[i] > 0.0:  # Quantification
                                    reward[i] = 1.0 + ndtw_reward
                                elif reward[i] < 0.0:
                                    reward[i] = -1.0 + ndtw_reward
                                else:
                                    raise NameError("The action doesn't change the move")
                                # Miss the target penalty
                                if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                    reward[i] -= (1.0 - last_dist[i]) * 2.0
                    print(f"step{t}: reward: {reward}, dist: {dist}, ndtw: {ndtw_score}")
                    rewards.append(reward)
                    rl_masks.append(rl_mask)
                    last_dist[:] = dist
                    last_ndtw[:] = ndtw_score

                    if self.args.step_wise_a2c:
                        discount_reward = rewards[t] # + discount_reward * self.args.gamma
                        rl_mask_ = torch.from_numpy(rl_masks[t]).cuda(device=model.device)
                        clip_reward = discount_reward.copy()
                        r_ = torch.from_numpy(clip_reward).cuda(device=hidden_states[t].device)
                        #v_ = model.critic(hidden_states[t].to(torch.float32))
                        #v_ = model.module.critic(hidden_states[t])
                        nav_inputs.update({'state': output['cls_hidden_state']})
                        v_ = model(mode='critic', batch=nav_inputs, state=hidden_states[t])
                        a_ = (r_ - v_).detach()

                        t_critic_loss = (((r_ - v_) ** 2)* rl_mask_).sum() * 0.5
                        t_policy_loss = (-policy_log_probs[t] * a_ * rl_mask_ ).sum()

                        rl_loss = t_policy_loss + t_critic_loss
                        if feedback == 'sample':
                            entropy_loss = (- self.args.entropy_loss_weight * entropys[t] * rl_mask_).sum()
                            rl_loss += entropy_loss

                        # Normalize the loss function
                        if self.args.normalize_loss == 'total':
                            rl_loss /= total
                        elif self.args.normalize_loss == 'batch':
                            rl_loss /= batch_size
                        else:
                            assert self.args.normalize_loss == 'none'

                        total = total + np.sum(rl_masks[t])

                        if not validate:
                            # rl_loss.backward()
                            cnt_loss += rl_loss
                            ml_loss += rl_loss.detach()
                            cnt_loss.backward()
                            cnt_loss = 0.

                            print(f"rank {args.rank}: rl loss backward!! step{t}: t_policy_loss:{t_policy_loss.detach()}, t_critic_loss: {t_critic_loss.detach()}, rl_loss: {rl_loss.detach()}, ml_loss: {ml_loss}")
                        rl_loss = 0.

                    # if self.args.cot_output_as_supervision:
                    #     for i in range(len(navigation_cot_gt)):
                    #         if reward[i] > 0:
                    #             new_cot_gt.append(batch_generated_sentences_navigation_cot[i])
                    #         else:
                    #             new_cot_gt.append(navigation_cot_gt[i])
                    #     nav_inputs.update({'new_cot_gt': new_cot_gt})
                    #     lm_logits = output['logits']
                    #     batch = collections.defaultdict(lambda: None, nav_inputs)
                    #     all_text = []
                    #     for bn in range(batch_size):
                    #         prompt = batch["prompts"][bn]
                    #
                    #         label = batch['new_cot_gt'][bn] + f"{model.module.lang_model.tokenizer.eos_token}"
                    #         # if self.args.check_cot_input_gt:
                    #         #     print(f"\n{prompt}{label}")
                    #         all_text.append([prompt, label])
                    #
                    #     text_input = model.module.lang_model.tokenize(all_text).to(batch['vp_img_embeds'].device)
                    #     labels = text_input['input_ids'].clone()
                    #     labels[text_input['token_type_ids'][:, -labels.shape[-1]:] == 0] = -100
                    #     shift_logits = lm_logits[..., :-1, :].contiguous()
                    #     shift_labels = labels[..., 1:].contiguous()
                    #     # Flatten the tokens
                    #     loss_fct = nn.CrossEntropyLoss()
                    #     shift_logits = shift_logits.view(-1, model.module.lang_model.config.vocab_size)
                    #     shift_labels = shift_labels.view(-1)
                    #     # Enable model parallelism
                    #     shift_labels = shift_labels.to(shift_logits.device)
                    #     lm_loss = loss_fct(shift_logits, shift_labels)
                    #
                    #     lm_loss = lm_loss * args.cotsum_loss_coef / batch_size / args.gradient_accumulation_step
                    #     if not flag:
                    #         # if self.args.add_lmloss_with_prob:
                    #         #     if random.random() < self.args.cal_lmloss_prob:
                    #         #         print("calculate lm loss!!!")
                    #         cnt_loss += lm_loss
                    #
                    #         ml_loss += cnt_loss.detach()
                    #         if not validate:
                    #             cnt_loss.backward()
                    #         cnt_loss = 0.
                    #         print(
                    #             f"rank {args.rank}: lm loss backward!! step{t}: lm_loss: {lm_loss.detach()}, ml_loss: {ml_loss}")

                ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # if self.args.self_improving_cot:
            #     if ended.all() or t == max_action_len - 1:
            #         flag = True

                if flag:
                    break

        if self.args.visualize_cot:
            visualize_file.write(str(visualize) + '\n')
            visualize_file.close()

        if enable_RL_A2C and not self.args.step_wise_a2c:
        #     discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
        #     for i in range(batch_size):
        #         if not ended[i]:  # If the action is not ended, use the value function as the last reward
        #             discount_reward[i] = last_value__[i]
        #
            length = len(rewards)
            total = 0
            rl_loss = 0.

            for rl_t in range(length-1, -1, -1):
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    #model.module.critic = model.module.critic.to(model.module.model_type)
                    # multi-gpu
                    if rl_t == 0:
                        context = nullcontext
                        #if enable_RL_A2C:
                        #critic_context = nullcontext
                    else:
                        context = model.no_sync
                        # if enable_RL_A2C:
                        #     critic_context = model.critic.no_sync
                        # else:
                        #     critic_context = nullcontext
                else:
                    #model.module.critic = model.module.critic.to(model.model_type)
                    # single-gpu
                    if rl_t == 0:
                        context = nullcontext
                        #if enable_RL_A2C:
                        # critic_context = nullcontext
                    else:
                        context = nullcontext
                        #if enable_RL_A2C:
                        # critic_context = nullcontext

                with context():
                    #with critic_context():

                    discount_reward = discount_reward * self.args.gamma + rewards[rl_t]  # If it ended, the reward will be 0
                    rl_mask_ = torch.from_numpy(rl_masks[rl_t]).cuda()
                    clip_reward = discount_reward.copy()
                    r_ = torch.from_numpy(clip_reward).cuda()
                    #v_ = model.module.critic(hidden_states[rl_t])
                    v_ = model(mode='critic', batch=nav_inputs, state=hidden_states[rl_t])
                    # v_ = model.critic(hidden_states[rl_t].to(torch.float32))
                    a_ = (r_ - v_).detach()

                    t_policy_loss = (-policy_log_probs[rl_t] * a_ * rl_mask_).sum()
                    t_critic_loss = (((r_ - v_) ** 2) * rl_mask_).sum() * 0.5 # 1/2 L2 loss

                    rl_loss += t_policy_loss + t_critic_loss
                    if feedback == 'sample':
                        entropy_loss = (- self.args.entropy_loss_weight * entropys[rl_t] * rl_mask_).sum()
                        rl_loss += entropy_loss
#                         # Normalize the loss function
#                         if self.args.normalize_loss == 'total':
#                             rl_loss /= total
#                         elif self.args.normalize_loss == 'batch':
#                             rl_loss /= batch_size
#                         else:
#                             assert self.args.normalize_loss == 'none'
#
# =                       rl_loss.backward()

                    # ml_loss += rl_loss.detach()
                    # print(f"rank {args.rank}: rl loss backward!! step{rl_t}: t_policy_loss:{t_policy_loss.detach()}, t_critic_loss: {t_critic_loss.detach()}, rl_loss: {rl_loss.detach()}, ml_loss: {ml_loss}")

                    total = total + np.sum(rl_masks[rl_t])

                    # rl_loss = 0.
                    #torch.cuda.empty_cache()


            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            #rl_loss.backward()
            #ml_loss += rl_loss.detach()
            cnt_loss += rl_loss
            ml_loss += cnt_loss.detach()
            if not validate:
                cnt_loss.backward()
            #cnt_loss = 0.
            print(f"rank {args.rank}: rl loss backward!! step{rl_t}: rl_loss: {rl_loss.detach()}, ml_loss: {ml_loss}")

        #torch.cuda.empty_cache()
        if self.args.enable_self_select:
            return ml_loss, self_select_loss, traj
        else:
            return ml_loss, traj

    def get_neg_cot(self, gt_cot, output_cot, direction_landmark_dict, direction_of_gt, data_type=None, nav_vpids=None, nav_targets=None, cand_masks=None, gmaps=None, obs=None, traj=None):

        bs = len(gt_cot)
        neg_cot = []

        # direction_gt_mapping = {
        #     "in front of":'go forward',
        #     "behind":'go back',
        #     "to the right of":'turn right',
        #     "to the left of":'turn left',
        #     "above":'go up',
        #     "below":'go down'
        # }

        direction_gt_mapping = {
            'go forward':"in front of",
            'go back':"behind",
            'turn right':"to the right of",
            'turn left':"to the left of",
            'go up':"above",
            'go down':"below"
        }

        if self.args.mlm:
            for bs_ind in range(bs):
                if self.args.landmark_not_merge_in_gt:
                    if nav_targets[bs_ind] == -100:
                        nav_target = 0
                    else:
                        nav_target = nav_targets[bs_ind]
                    ob = obs[bs_ind]

                    cand_masks[bs_ind][nav_target] = False
                    navigable_wogt_indices = torch.nonzero(cand_masks[bs_ind]).squeeze().cpu().numpy().tolist()
                    try:
                        neg_target = random.sample(navigable_wogt_indices, 1)[0]
                    except:
                        if self.args.mlm:
                            neg_cot.append("I should go back to an observation with room.")
                        else:
                            neg_cot.append("I should go to an observation with room behind me.")

                        continue

                    # common sense
                    #if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie':
                    if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie' or \
                            data_type[bs_ind] == 'cvdn' or  data_type[bs_ind] == 'soon':
                        neg_vpid = nav_vpids[bs_ind][neg_target]  ## Use a negative target!!
                        if neg_vpid is not None:
                            gt_sub_path = gmaps[bs_ind].graph.path(ob['viewpoint'], neg_vpid)
                            if len(gt_sub_path) == 1:
                                prev_vp = traj[bs_ind]['path'][-1][-1]
                            else:
                                prev_vp = gt_sub_path[-2]
                            viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][neg_vpid]
                            # gt_landmarks = load_json(
                            #     os.path.join(self.t2t_landmark_dir, ob['scan'], neg_vpid,
                            #                  f"{ob['scan']}_{neg_vpid}.json"))[
                            #     str(viewidx)]
                            if self.args.cot_summarization:
                                direction_of_gt = self.get_direction_vp(
                                    gmaps[bs_ind].node_positions[obs[bs_ind]['viewpoint']],
                                    gmaps[bs_ind].node_positions[neg_vpid],
                                    obs[bs_ind]['heading'], obs[bs_ind]['elevation'],
                                    spatial_relation=True)
                                # action_of_gt = ' '.join(
                                #     self.get_direction_vp(gmaps[bs_ind].node_positions[obs[bs_ind]['viewpoint']],
                                #                           gmaps[bs_ind].node_positions[neg_vpid], obs[bs_ind]['heading'],
                                #                           obs[bs_ind]['elevation']).split(' ')[:-1])
                                gt_landmarks = load_json(
                                    os.path.join(self.t2t_landmark_dir, ob['scan'], neg_vpid,
                                                 f"{ob['scan']}_{neg_vpid}.json"))[str(viewidx)]
                                if len(gt_landmarks) > self.args.land_num:
                                    gt_landmarks = random.sample(gt_landmarks, self.args.land_num)

                                if self.args.cot_v4_only_direction:
                                    common_sense = f"I should go to an observation {direction_of_gt} me."
                                elif self.args.cot_v4_only_landmark:
                                    common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}]."
                                else:
                                    if self.args.mlm:
                                        common_sense = f"I should {direction_of_gt} to an observation with {', '.join(gt_landmarks)}."
                                    else:
                                        common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                        else:
                            if self.args.mlm:
                                common_sense = f"I should stop at to an observation."
                            else:
                                common_sense = f"Observation matches with long-term goal, so stop."

                        if self.args.cot_v4:
                            common_sense = f"{common_sense}"
                        neg_cot.append(common_sense)

                else:
                    direction_keys = list(direction_landmark_dict.keys())
                    if direction_of_gt not in direction_keys:
                        replace_direction = random.choice(direction_keys)
                        replace_landmark = direction_landmark_dict[replace_direction]
                        # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                        modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."
                        neg_cot.append(modify_output_cot)
                    else:
                        direction_keys = list(direction_landmark_dict.keys())
                        direction_keys.remove(direction_of_gt)
                        replace_direction = random.choice(direction_keys)
                        replace_landmark = direction_landmark_dict[replace_direction]
                        # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                        modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."
                        neg_cot.append(modify_output_cot)

        elif self.args.cot_output_as_supervision:
            for bs_ind in range(bs):
                if self.args.landmark_not_merge_in_gt:
                    if nav_targets[bs_ind] == -100:
                        nav_target = 0
                    else:
                        nav_target = nav_targets[bs_ind]
                    ob = obs[bs_ind]

                    cand_masks[bs_ind][nav_target] = False
                    navigable_wogt_indices = torch.nonzero(cand_masks[bs_ind]).squeeze().cpu().numpy().tolist()
                    try:
                        neg_target = random.sample(navigable_wogt_indices, 1)[0]
                    except:
                        if self.args.mlm:
                            neg_cot.append("I should go back to an observation with room.")
                        else:
                            neg_cot.append("I should go to an observation with room behind me.")

                        continue

                    # common sense
                    #if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie':
                        # if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie':
                    if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie' or \
                                data_type[bs_ind] == 'cvdn' or data_type[bs_ind] == 'soon':
                        neg_vpid = nav_vpids[bs_ind][neg_target]  ## Use a negative target!!
                        if neg_vpid is not None:
                            gt_sub_path = gmaps[bs_ind].graph.path(ob['viewpoint'], neg_vpid)
                            if len(gt_sub_path) == 1:
                                prev_vp = traj[bs_ind]['path'][-1][-1]
                            else:
                                prev_vp = gt_sub_path[-2]
                            viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][neg_vpid]
                            #gt_landmarks = load_json(
                            #    os.path.join(self.t2t_landmark_dir, ob['scan'], neg_vpid,
                            #                 f"{ob['scan']}_{neg_vpid}.json"))[
                            #    str(viewidx)]
                            if self.args.cot_summarization:
                                direction_of_gt = self.get_direction_vp(
                                    gmaps[bs_ind].node_positions[obs[bs_ind]['viewpoint']],
                                    gmaps[bs_ind].node_positions[neg_vpid],
                                    obs[bs_ind]['heading'], obs[bs_ind]['elevation'],
                                    spatial_relation=True)
                                # action_of_gt = ' '.join(
                                #     self.get_direction_vp(gmaps[bs_ind].node_positions[obs[bs_ind]['viewpoint']],
                                #                           gmaps[bs_ind].node_positions[neg_vpid], obs[bs_ind]['heading'],
                                #                           obs[bs_ind]['elevation']).split(' ')[:-1])
                                gt_landmarks = load_json(
                                    os.path.join(self.t2t_landmark_dir, ob['scan'], neg_vpid,
                                                 f"{ob['scan']}_{neg_vpid}.json"))[str(viewidx)]
                                if len(gt_landmarks) > self.args.land_num:
                                    gt_landmarks = random.sample(gt_landmarks, self.args.land_num)
                                print('gt_landmarks')
                                print(gt_landmarks)
                                print('direction_of_gt')
                                print(direction_of_gt)
                                if self.args.cot_v4_only_direction:
                                    common_sense = f"I should go to an observation {direction_of_gt} me."
                                elif self.args.cot_v4_only_landmark:
                                    common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}]."
                                else:
                                    if self.args.mlm:
                                        common_sense = f"I should {direction_of_gt} to an observation with {', '.join(gt_landmarks)}."
                                    else:
                                        common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                        else:
                            if self.args.mlm:
                                common_sense = f"I should stop at to an observation."
                            else:
                                common_sense = f"Observation matches with long-term goal, so stop."

                        if self.args.cot_v4:
                            common_sense = f"{common_sense}"
                        neg_cot.append(common_sense)

                else:
                    direction_keys = list(direction_landmark_dict.keys())
                    if direction_of_gt not in direction_keys:
                        replace_direction = random.choice(direction_keys)
                        replace_landmark = direction_landmark_dict[replace_direction]
                        # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {replace_direction} me."
                        # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                        modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."

                        neg_cot.append(modify_output_cot)
                    else:
                        direction_keys = list(direction_landmark_dict.keys())
                        direction_keys.remove(direction_of_gt)
                        replace_direction = random.choice(direction_keys)
                        replace_landmark = direction_landmark_dict[replace_direction]
                        # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {replace_direction} me."
                        # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                        modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."

                        neg_cot.append(modify_output_cot)
        else:
            for bs_ind in range(bs):
                direction_keys = list(direction_landmark_dict.keys())
                if direction_of_gt not in direction_keys:
                    replace_direction = random.choice(direction_keys)
                    replace_landmark = direction_landmark_dict[replace_direction]
                    # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                    modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."
                    neg_cot.append(modify_output_cot)
                else:
                    direction_keys = list(direction_landmark_dict.keys())
                    direction_keys.remove(direction_of_gt)
                    replace_direction = random.choice(direction_keys)
                    replace_landmark = direction_landmark_dict[replace_direction]
                    # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                    modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."
                    neg_cot.append(modify_output_cot)
                # if output_cot[bs_ind] == gt_cot[bs_ind]:
                #     if self.args.landmark_not_merge_in_gt:
                #         if nav_targets[bs_ind] == -100:
                #             nav_target = 0
                #         else:
                #             nav_target = nav_targets[bs_ind]
                #         ob = obs[bs_ind]
                #
                #         cand_masks[bs_ind][nav_target] = False
                #         navigable_wogt_indices = torch.nonzero(cand_masks[bs_ind]).squeeze().cpu().numpy().tolist()
                #         try:
                #             neg_target = random.sample(navigable_wogt_indices,1)[0]
                #         except:
                #             if self.args.mlm:
                #                 neg_cot.append("I should go back to an observation with room.")
                #             else:
                #                 neg_cot.append("I should go to an observation with room behind me.")
                #
                #             continue
                #
                #         # common sense
                #         #if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie':
                #             # if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie':
                #         if data_type[bs_ind] == 'r2r' or data_type[bs_ind] == 'reverie' or \
                #                     data_type[bs_ind] == 'cvdn' or data_type[bs_ind] == 'soon':
                #             neg_vpid = nav_vpids[bs_ind][neg_target] ## Use a negative target!!
                #             if neg_vpid is not None:
                #                 gt_sub_path = gmaps[bs_ind].graph.path(ob['viewpoint'], neg_vpid)
                #                 if len(gt_sub_path) == 1:
                #                     prev_vp = traj[bs_ind]['path'][-1][-1]
                #                 else:
                #                     prev_vp = gt_sub_path[-2]
                #                 viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][neg_vpid]
                #                 gt_landmarks = load_json(
                #                     os.path.join(self.t2t_landmark_dir, ob['scan'], neg_vpid,
                #                                  f"{ob['scan']}_{neg_vpid}.json"))[
                #                     str(viewidx)]
                #                 if self.args.cot_summarization:
                #                     direction_of_gt = self.get_direction_vp(gmaps[bs_ind].node_positions[obs[bs_ind]['viewpoint']],
                #                                                             gmaps[bs_ind].node_positions[neg_vpid],
                #                                                             obs[bs_ind]['heading'], obs[bs_ind]['elevation'],
                #                                                             spatial_relation=True)
                #                     # action_of_gt = ' '.join(
                #                     #     self.get_direction_vp(gmaps[bs_ind].node_positions[obs[bs_ind]['viewpoint']],
                #                     #                           gmaps[bs_ind].node_positions[neg_vpid], obs[bs_ind]['heading'],
                #                     #                           obs[bs_ind]['elevation']).split(' ')[:-1])
                #                     gt_landmarks = load_json(
                #                         os.path.join(self.t2t_landmark_dir, ob['scan'], neg_vpid,
                #                                      f"{ob['scan']}_{neg_vpid}.json"))[str(viewidx)]
                #                     if len(gt_landmarks) > 5:
                #                         gt_landmarks = random.sample(gt_landmarks, 5)
                #
                #                     if self.args.cot_v4_only_direction:
                #                         common_sense = f"I should go to an observation {direction_of_gt} me."
                #                     else:
                #                         if self.args.mlm:
                #                             common_sense = f"I should {direction_of_gt} to an observation with {', '.join(gt_landmarks)}."
                #                         else:
                #                             common_sense = f"I should go to an observation with [{', '.join(gt_landmarks)}] {direction_of_gt} me."
                #             else:
                #                 if self.args.mlm:
                #                     common_sense = f"I should stop at to an observation."
                #                 else:
                #                     common_sense = f"Observation matches with long-term goal, so stop."
                #
                #             if self.args.cot_v4:
                #                 common_sense = f"{common_sense}"
                #             neg_cot.append(common_sense)
                #
                #     else:
                #         direction_keys = list(direction_landmark_dict.keys())
                #         if direction_of_gt not in direction_keys:
                #             replace_direction = random.choice(direction_keys)
                #             replace_landmark = direction_landmark_dict[replace_direction]
                #             # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                #             modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."
                #             neg_cot.append(modify_output_cot)
                #         else:
                #             direction_keys = list(direction_landmark_dict.keys())
                #             direction_keys.remove(direction_of_gt)
                #             replace_direction = random.choice(direction_keys)
                #             replace_landmark = direction_landmark_dict[replace_direction]
                #             # modify_output_cot = f"I should go to an observation with [{', '.join(replace_landmark)}] {direction_gt_mapping[replace_direction]} me."
                #             modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."
                #             neg_cot.append(modify_output_cot)
                # else:
                #     print("output do not match gt!!!")
                #     neg_cot.append(output_cot[bs_ind])


        return neg_cot


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
        elif mode == "navigation_once_forward_cot_navigation":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                if self.args.mlm:
                    prompts.append(
                        self.get_prompt(
                            "navigation_once_forward_cot_navigation",
                            instruction=batch["instruction"][bn],
                            hist_num=hist_nums[bn],
                            cand_num=cand_nums[bn],
                            cand_landmarks=batch["vp_landmarks"][bn],
                            nav_vpids=batch['gmap_vpids'][bn],
                            cand_masks=cand_masks[bn],
                            cls_token=kwargs.get("cls_token"),
                            land_token=kwargs.get("land_token"),
                            dir_token=kwargs.get("dir_token"),
                        )
                    )
                else:
                    prompts.append(
                        self.get_prompt(
                            "navigation_once_forward_cot_navigation",
                            instruction=batch["instruction"][bn],
                            hist_num=hist_nums[bn],
                            cand_num=cand_nums[bn],
                            cand_landmarks=batch["vp_landmarks"][bn],
                            nav_vpids=batch['gmap_vpids'][bn],
                            cand_masks=cand_masks[bn],
                            cls_token=kwargs.get("cls_token"),
                        )
                    )
        elif mode == "navigation_self_refine":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation_self_refine",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cand_landmarks=batch["vp_landmarks"][bn],
                        nav_vpids=batch['gmap_vpids'][bn],
                        cand_masks=cand_masks[bn],
                        neg_cot=batch['navigation_self_refine_neg'][bn]
                    )
                )
        elif mode == "navigation_self_select":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation_self_select",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cand_landmarks=batch["vp_landmarks"][bn],
                        nav_vpids=batch['gmap_vpids'][bn],
                        cand_masks=cand_masks[bn],
                        cot_pair=batch['navigation_cot_pair'][bn]
                    )
                )

        else:
            raise NotImplementedError

        return prompts
