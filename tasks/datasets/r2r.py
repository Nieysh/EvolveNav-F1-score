import json
import base64
import os
import re
import shutil
import urllib.request
import urllib.error
import numpy as np
from .mp3d_dataset import MP3DDataset
from collections import defaultdict
ERROR_MARGIN = 3.0


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _normalize_direction(text):
    text = _normalize_text(text)
    direction_aliases = [
        ("front", ["in front of", "go forward", "forward", "front"]),
        ("back", ["behind", "go back", "back", "rear"]),
        ("right", ["to the right of", "turn right", "right"]),
        ("left", ["to the left of", "turn left", "left"]),
        ("up", ["above", "go up", "upon", "up"]),
        ("down", ["below", "go down", "under", "down"]),
        ("stop", ["stop"]),
    ]
    for direction, aliases in direction_aliases:
        if any(alias in text for alias in aliases):
            return direction
    return ""


def _extract_predicted_landmarks(reasoning):
    reasoning = str(reasoning)
    bracketed = re.findall(r"\[([^\]]+)\]", reasoning)
    if bracketed:
        landmarks = []
        for group in bracketed:
            landmarks.extend([x.strip().lower() for x in group.split(",") if x.strip()])
        return landmarks

    match = re.search(r"with\s+(.+?)(?:\s+(?:in front of|behind|to the right of|to the left of|above|below)\s+me|\.|$)", reasoning, re.I)
    if not match:
        return []
    return [x.strip().lower() for x in match.group(1).split(",") if x.strip()]


def _landmark_in_gt(pred_landmark, gt_landmarks):
    pred_landmark = _normalize_text(pred_landmark)
    gt_landmarks = [_normalize_text(x) for x in gt_landmarks]
    return any(pred_landmark == gt or pred_landmark in gt or gt in pred_landmark for gt in gt_landmarks)


def _safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", str(text), re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value

class R2RDataset(MP3DDataset):
    name = "r2r"

    def load_data(self, anno_file, max_instr_len=200, debug=False, visualize=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """
        with open(str(anno_file), "r") as f:
            data = json.load(f)
        new_data = []
        sample_index = 0

        for i, item in enumerate(data):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['raw_idx'] = i
                new_item['sample_idx'] = sample_index
                new_item['instr_id'] = 'r2r_{}_{}'.format(item['path_id'], j)

                new_item['instruction'] = instr
                del new_item['instructions']

                if 'instr_encodings' in new_item:
                    new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                    del new_item['instr_encodings']

                if 'new_instructions' in new_item and len(eval(item['new_instructions'])) > j:
                    new_item['fg_instruction'] = eval(item['new_instructions'])[j]
                    new_item['fg_instruction'] = [' '.join(instr) for instr in new_item['fg_instruction']]
                    del new_item['new_instructions']
                    new_item['fg_view'] = item['chunk_view'][j]
                    fg_view = []
                    for idx, index in enumerate(new_item['fg_view']):
                        index_num = index[1] - index[0]
                        fg_view += [idx] * index_num
                    new_item['fg_view'] = fg_view
                    del new_item['chunk_view']

                new_item['data_type'] = 'r2r'
                new_data.append(new_item)
                sample_index += 1

        if debug:
            new_data = new_data[:20]
        if visualize:
            new_data = new_data[:20]
        if getattr(self.args, "eval_episode_limit", None) is not None and not self.training:
            new_data = new_data[:self.args.eval_episode_limit]
        #new_data = new_data[:20]
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }
        return new_data, gt_trajs


    def eval_metrics(self, preds, logger, name):
        """
        Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]
        :param preds:
        :param logger:
        :param name:
        :return:
        """
        logger.info('eval %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        action_reasoning_enabled = (
            getattr(self.args, "enable_action_reasoning_f1", False)
            and getattr(self.args, "action_reasoning_eval_phase", "both") == "both"
        )
        action_reasoning_total_steps = 0
        if action_reasoning_enabled:
            action_reasoning_total_steps = sum(
                len(item.get('gt_node', {})) for item in preds
            )
            logger.info(
                "[ActionReasoningF1] start judge_mode=%s predictions=%d judged_steps=%d",
                getattr(self.args, "action_reasoning_judge_mode", "text"),
                len(preds),
                action_reasoning_total_steps,
            )

        judged_steps = 0
        for item_idx, item in enumerate(preds, start=1):
            instr_id = item['instr_id']
            traj = item['trajectory']

            if instr_id not in self.gt_trajs.keys():
                print("instr_id {} not in self.gt_trajs".format(instr_id))
                raise NotImplementedError

            if name == "R2R":
                scan, gt_traj = self.gt_trajs[instr_id]
                traj_scores = self.eval_dis_item(scan, traj, gt_traj)
            else:
                raise NotImplementedError

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

            if action_reasoning_enabled:
                ar_scores, judged_steps = self.eval_action_reasoning_item(
                    item,
                    logger=logger,
                    item_idx=item_idx,
                    total_items=len(preds),
                    judged_steps=judged_steps,
                    total_steps=action_reasoning_total_steps,
                )
                for k, v in ar_scores.items():
                    metrics[k].append(v)

        if name in ['R2R']:
            avg_metrics = {
                'action_steps': np.mean(metrics['action_steps']),
                'steps': np.mean(metrics['trajectory_steps']),
                'lengths': np.mean(metrics['trajectory_lengths']),
                'nav_error': np.mean(metrics['nav_error']),
                'oracle_error': np.mean(metrics['oracle_error']),
                'sr': np.mean(metrics['success']) * 100,
                'oracle_sr': np.mean(metrics['oracle_success']) * 100,
                'spl': np.mean(metrics['spl']) * 100,
            }
            if action_reasoning_enabled:
                action_reasoning_counts = {
                    'tp': int(np.sum(metrics['action_reasoning_tp'])),
                    'fp': int(np.sum(metrics['action_reasoning_fp'])),
                    'fn': int(np.sum(metrics['action_reasoning_fn'])),
                    'tn': int(np.sum(metrics['action_reasoning_tn'])),
                }
                precision_den = action_reasoning_counts['tp'] + action_reasoning_counts['fp']
                recall_den = action_reasoning_counts['tp'] + action_reasoning_counts['fn']
                precision = action_reasoning_counts['tp'] / precision_den if precision_den > 0 else 0.0
                recall = action_reasoning_counts['tp'] / recall_den if recall_den > 0 else 0.0
                f1_den = precision + recall
                avg_metrics.update({
                    'action_reasoning_precision': precision * 100,
                    'action_reasoning_recall': recall * 100,
                    'action_reasoning_f1': (2 * precision * recall / f1_den * 100) if f1_den > 0 else 0.0,
                    'action_reasoning_tp': action_reasoning_counts['tp'],
                    'action_reasoning_fp': action_reasoning_counts['fp'],
                    'action_reasoning_fn': action_reasoning_counts['fn'],
                    'action_reasoning_tn': action_reasoning_counts['tn'],
                })
                logger.info(
                    "[ActionReasoningF1] TP=%d FP=%d FN=%d TN=%d P=%.2f R=%.2f F1=%.2f",
                    action_reasoning_counts['tp'], action_reasoning_counts['fp'],
                    action_reasoning_counts['fn'], action_reasoning_counts['tn'],
                    avg_metrics['action_reasoning_precision'],
                    avg_metrics['action_reasoning_recall'],
                    avg_metrics['action_reasoning_f1'],
                )
        else:
            raise NotImplementedError
        return avg_metrics, metrics

    def eval_action_reasoning_item(self, item, logger=None, item_idx=None, total_items=None,
                                   judged_steps=0, total_steps=0):
        scores = {
            'action_reasoning_tp': 0,
            'action_reasoning_fp': 0,
            'action_reasoning_fn': 0,
            'action_reasoning_tn': 0,
        }
        pred_cots = item.get('generated_sentences_navigation_cot', {})
        pred_nodes = item.get('pred_node', {})
        gt_nodes = item.get('gt_node', {})
        gt_directions = item.get('direction_of_gt', {})
        gt_landmarks = item.get('gt_landmarks', {})
        gt_vpids = item.get('gt_vpid', {})
        gt_action_viewpoints = item.get('gt_action_viewpoint', {})
        gt_viewidxs = item.get('gt_viewidx', {})
        step_details = {}
        progress_every = max(1, int(getattr(self.args, "action_reasoning_progress_every", 10)))
        if logger is not None:
            logger.info(
                "[ActionReasoningF1] item %d/%d instr_id=%s steps=%d",
                item_idx,
                total_items,
                item.get('instr_id', ''),
                len(gt_nodes),
            )

        for step in sorted(gt_nodes.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            pred_node = pred_nodes.get(step)
            gt_node = gt_nodes.get(step)
            action_correct = pred_node == gt_node
            reasoning = pred_cots.get(step, "")
            reasoning_correct, reason_detail = self.judge_action_reasoning(
                reasoning,
                gt_directions.get(step, ""),
                gt_landmarks.get(step, []),
                gt_node,
                item,
                step,
                gt_vpids.get(step),
                gt_action_viewpoints.get(step),
                gt_viewidxs.get(step),
            )

            if action_correct and reasoning_correct:
                scores['action_reasoning_tp'] += 1
                label = 'TP'
            elif (not action_correct) and reasoning_correct:
                scores['action_reasoning_fp'] += 1
                label = 'FP'
            elif action_correct and (not reasoning_correct):
                scores['action_reasoning_fn'] += 1
                label = 'FN'
            else:
                scores['action_reasoning_tn'] += 1
                label = 'TN'

            step_details[step] = {
                'label': label,
                'action_correct': action_correct,
                'reasoning_correct': reasoning_correct,
                **reason_detail,
            }
            judged_steps += 1
            if logger is not None and (
                    judged_steps == 1 or judged_steps == total_steps or judged_steps % progress_every == 0):
                logger.info(
                    "[ActionReasoningF1] progress judged_steps=%d/%d item=%d/%d instr_id=%s step=%s label=%s action_correct=%s reasoning_correct=%s image=%s",
                    judged_steps,
                    total_steps,
                    item_idx,
                    total_items,
                    item.get('instr_id', ''),
                    step,
                    label,
                    action_correct,
                    reasoning_correct,
                    reason_detail.get('judge_image_path', ''),
                )

        item['action_reasoning_eval'] = step_details
        return scores, judged_steps

    def export_action_reasoning_facts(self, preds, path, logger=None, name="R2R"):
        path_dir = os.path.dirname(path)
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)
        facts = []
        total_steps = sum(len(item.get('gt_node', {})) for item in preds)
        progress_every = max(1, int(getattr(self.args, "action_reasoning_progress_every", 10)))
        exported = 0
        missing_images = 0
        missing_reasoning = 0
        if logger is not None:
            logger.info(
                "[ActionReasoningF1] export facts start predictions=%d steps=%d path=%s",
                len(preds), total_steps, path,
            )

        for item_idx, item in enumerate(preds, start=1):
            pred_cots = item.get('generated_sentences_navigation_cot', {})
            pred_nodes = item.get('pred_node', {})
            gt_nodes = item.get('gt_node', {})
            gt_directions = item.get('direction_of_gt', {})
            gt_landmarks = item.get('gt_landmarks', {})
            gt_vpids = item.get('gt_vpid', {})
            pred_vpids = item.get('pred_vpid', {})
            gt_action_viewpoints = item.get('gt_action_viewpoint', {})
            gt_viewidxs = item.get('gt_viewidx', {})
            navigation_cot_gt = item.get('navigation_cot_gt', {})
            prompts = item.get('prompts', {})

            for step in sorted(gt_nodes.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                reasoning = pred_cots.get(step, "")
                if not _normalize_text(reasoning):
                    missing_reasoning += 1
                gt_node = gt_nodes.get(step)
                pred_node = pred_nodes.get(step)
                gt_direction_norm = _normalize_direction(gt_directions.get(step, ""))
                pred_direction_norm = _normalize_direction(reasoning)
                reasoning_text = _normalize_text(reasoning)
                if gt_node == 0:
                    direction_correct = pred_direction_norm == "stop" or "stop" in reasoning_text
                    source_image_path = None
                    judge_image_path = None
                    judge_image_relpath = None
                    judge_skipped = "stop_action"
                else:
                    direction_correct = bool(gt_direction_norm) and pred_direction_norm == gt_direction_norm
                    source_image_path = self.resolve_action_reasoning_image(
                        item,
                        gt_action_viewpoints.get(step),
                        gt_viewidxs.get(step),
                    )
                    judge_image_path = None
                    judge_image_relpath = None
                    judge_skipped = None
                    if source_image_path is None:
                        missing_images += 1
                    else:
                        judge_image_path = self.keep_action_reasoning_judge_image(
                            image_path=source_image_path,
                            item=item,
                            step=step,
                            gt_action_viewpoint=gt_action_viewpoints.get(step),
                            gt_viewidx=gt_viewidxs.get(step),
                        )
                        judge_image_relpath = os.path.relpath(judge_image_path, self.args.output_dir)

                fact = {
                    "dataset": name,
                    "instr_id": item.get("instr_id", ""),
                    "scan": item.get("scan", ""),
                    "step": str(step),
                    "action_correct": pred_node == gt_node,
                    "pred_node": pred_node,
                    "gt_node": gt_node,
                    "pred_vpid": pred_vpids.get(step),
                    "gt_vpid": gt_vpids.get(step),
                    "gt_action_viewpoint": gt_action_viewpoints.get(step),
                    "gt_viewidx": gt_viewidxs.get(step),
                    "reasoning": reasoning,
                    "navigation_cot_gt": navigation_cot_gt.get(step, ""),
                    "prompt": prompts.get(step, ""),
                    "gt_direction": gt_direction_norm,
                    "pred_direction": pred_direction_norm,
                    "direction_correct": direction_correct,
                    "predicted_landmarks": _extract_predicted_landmarks(reasoning),
                    "gt_landmarks": gt_landmarks.get(step, []),
                    "source_image_path": source_image_path,
                    "judge_image_path": judge_image_path,
                    "judge_image_relpath": judge_image_relpath,
                    "judge_skipped": judge_skipped,
                }
                facts.append(fact)
                exported += 1
                if logger is not None and (
                        exported == 1 or exported == total_steps or exported % progress_every == 0):
                    logger.info(
                        "[ActionReasoningF1] export progress steps=%d/%d item=%d/%d instr_id=%s step=%s image=%s",
                        exported, total_steps, item_idx, len(preds), item.get("instr_id", ""), step,
                        judge_image_path or "",
                    )

        payload = {
            "version": 1,
            "description": "Offline action-reasoning-F1 facts. Run scripts/evaluation/compute_action_reasoning_f1_from_facts.py on an internet-connected machine.",
            "output_dir": self.args.output_dir,
            "judge_image_dir": os.path.join(self.args.output_dir, "action_reasoning_judge_images"),
            "num_predictions": len(preds),
            "num_steps": len(facts),
            "missing_image_steps": missing_images,
            "missing_reasoning_steps": missing_reasoning,
            "facts": facts,
        }
        with open(path, "w", encoding="utf-8") as fout:
            json.dump(_json_safe(payload), fout, indent=2)
        if logger is not None:
            logger.info(
                "[ActionReasoningF1] exported facts=%d missing_images=%d missing_reasoning=%d to %s",
                len(facts), missing_images, missing_reasoning, path,
            )
        return path

    def judge_action_reasoning(self, reasoning, gt_direction, gt_landmarks, gt_node,
                               item=None, step=None, gt_vpid=None, gt_action_viewpoint=None, gt_viewidx=None):
        reasoning_text = _normalize_text(reasoning)
        gt_direction_norm = _normalize_direction(gt_direction)
        pred_direction_norm = _normalize_direction(reasoning)
        judge_mode = getattr(self.args, "action_reasoning_judge_mode", "text")

        if gt_node == 0:
            direction_correct = pred_direction_norm == 'stop' or 'stop' in reasoning_text
            landmark_correct = True
            landmark_detail = {'judge_mode': judge_mode, 'judge_skipped': 'stop_action'}
        else:
            direction_correct = bool(gt_direction_norm) and pred_direction_norm == gt_direction_norm
            predicted_landmarks = _extract_predicted_landmarks(reasoning)
            if judge_mode == "vlm":
                landmark_correct, landmark_detail = self.vlm_judge_landmarks(
                    reasoning=reasoning,
                    predicted_landmarks=predicted_landmarks,
                    item=item,
                    step=step,
                    gt_vpid=gt_vpid,
                    gt_action_viewpoint=gt_action_viewpoint,
                    gt_viewidx=gt_viewidx,
                )
            else:
                gt_landmarks = gt_landmarks or []
                if predicted_landmarks:
                    landmark_correct = all(_landmark_in_gt(x, gt_landmarks) for x in predicted_landmarks)
                else:
                    landmark_correct = any(_landmark_in_gt(x, [reasoning_text]) for x in gt_landmarks)
                landmark_detail = {
                    'judge_mode': 'text',
                    'predicted_landmarks': predicted_landmarks,
                    'gt_landmarks': gt_landmarks,
                }

        return direction_correct and landmark_correct, {
            'pred_direction': pred_direction_norm,
            'gt_direction': gt_direction_norm,
            'direction_correct': direction_correct,
            'landmark_correct': landmark_correct,
            **landmark_detail,
        }

    def vlm_judge_landmarks(self, reasoning, predicted_landmarks, item, step,
                            gt_vpid, gt_action_viewpoint, gt_viewidx):
        image_path = self.resolve_action_reasoning_image(item, gt_action_viewpoint, gt_viewidx)
        if image_path is None:
            raise RuntimeError(
                "VLM action-reasoning judge needs a GT action-view image. "
                "Set --action_reasoning_image_dir/--action_reasoning_image_pattern "
                "or --action_reasoning_scan_dir for MatterSim rendering."
            )
        judge_image_path = self.keep_action_reasoning_judge_image(
            image_path=image_path,
            item=item,
            step=step,
            gt_action_viewpoint=gt_action_viewpoint,
            gt_viewidx=gt_viewidx,
        )
        api_key_env = getattr(self.args, "action_reasoning_vlm_api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing VLM API key env var: {api_key_env}")

        with open(judge_image_path, "rb") as fin:
            image_b64 = base64.b64encode(fin.read()).decode("utf-8")

        prompt = (
            "You are judging one navigation reasoning step. The image is the ground-truth action view. "
            "Decide whether the landmarks/objects/places mentioned in the model reasoning are visible "
            "in the image, allowing reasonable synonyms and visually equivalent descriptions. "
            "Do not require exact wording. If the reasoning mentions no visual landmark/object/place, answer false. "
            "Return only JSON with keys: landmarks_visible (boolean), visible_evidence (list of strings), "
            "missing_or_uncertain (list of strings), explanation (short string).\n\n"
            f"Model reasoning: {reasoning}\n"
            f"Parsed landmark phrases: {predicted_landmarks}\n"
            f"GT next viewpoint id: {gt_vpid}\n"
            f"Instruction id: {item.get('instr_id', '') if item else ''}, step: {step}"
        )

        payload = {
            "model": getattr(self.args, "action_reasoning_vlm_model", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256,
        }
        req = urllib.request.Request(
            getattr(self.args, "action_reasoning_vlm_base_url", "https://api.openai.com/v1/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"VLM judge HTTP error {exc.code}: {error_body}") from exc

        content = response["choices"][0]["message"]["content"]
        judge = _safe_json_loads(content)
        return bool(judge.get("landmarks_visible", False)), {
            "judge_mode": "vlm",
            "source_image_path": image_path,
            "judge_image_path": judge_image_path,
            "predicted_landmarks": predicted_landmarks,
            "vlm_judge_raw": judge,
        }

    def keep_action_reasoning_judge_image(self, image_path, item, step, gt_action_viewpoint, gt_viewidx):
        out_dir = os.path.join(self.args.output_dir, "action_reasoning_judge_images")
        os.makedirs(out_dir, exist_ok=True)
        instr_id = _normalize_text(item.get("instr_id", "unknown") if item else "unknown")
        instr_id = re.sub(r"[^a-z0-9_.-]+", "_", instr_id).strip("_") or "unknown"
        scan = _normalize_text(item.get("scan", "unknown") if item else "unknown")
        scan = re.sub(r"[^a-z0-9_.-]+", "_", scan).strip("_") or "unknown"
        ext = os.path.splitext(image_path)[1] or ".jpg"
        filename = f"{instr_id}_step{step}_{scan}_{gt_action_viewpoint}_{gt_viewidx}{ext}"
        dst = os.path.join(out_dir, filename)
        if os.path.abspath(image_path) != os.path.abspath(dst):
            shutil.copy2(image_path, dst)
        return dst

    def resolve_action_reasoning_image(self, item, action_viewpoint, gt_viewidx):
        if action_viewpoint is None or gt_viewidx is None:
            return None
        scan = item.get("scan", "") if item else ""
        image_dir = getattr(self.args, "action_reasoning_image_dir", None)
        pattern = getattr(self.args, "action_reasoning_image_pattern", None)
        if image_dir:
            candidates = []
            if pattern:
                candidates.append(os.path.join(image_dir, pattern.format(
                    scan=scan, viewpoint=action_viewpoint, viewidx=gt_viewidx,
                )))
            else:
                candidates.extend([
                    os.path.join(image_dir, scan, action_viewpoint, f"{gt_viewidx}.jpg"),
                    os.path.join(image_dir, scan, action_viewpoint, f"{gt_viewidx}.png"),
                    os.path.join(image_dir, scan, f"{action_viewpoint}_{gt_viewidx}.jpg"),
                    os.path.join(image_dir, scan, f"{action_viewpoint}_{gt_viewidx}.png"),
                    os.path.join(image_dir, f"{scan}_{action_viewpoint}_{gt_viewidx}.jpg"),
                    os.path.join(image_dir, f"{scan}_{action_viewpoint}_{gt_viewidx}.png"),
                ])
            for path in candidates:
                if os.path.exists(path):
                    return path

        scan_dir = getattr(self.args, "action_reasoning_scan_dir", None)
        if scan_dir:
            return self.render_action_reasoning_image(scan, action_viewpoint, int(gt_viewidx), scan_dir)
        return None

    def render_action_reasoning_image(self, scan, viewpoint, viewidx, scan_dir):
        import math
        from PIL import Image
        import MatterSim

        cache_dir = getattr(self.args, "action_reasoning_image_cache_dir", None)
        if cache_dir is None:
            cache_dir = os.path.join(self.args.output_dir, "action_reasoning_images")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, f"{scan}_{viewpoint}_{viewidx}.jpg")
        if os.path.exists(out_path):
            return out_path

        sim = MatterSim.Simulator()
        sim.setNavGraphPath(self.connectivity_dir)
        sim.setDatasetPath(scan_dir)
        sim.setRenderingEnabled(True)
        sim.setCameraResolution(640, 480)
        sim.setCameraVFOV(math.radians(60))
        sim.setDiscretizedViewingAngles(True)
        sim.setDepthEnabled(False)
        sim.setPreloadingEnabled(False)
        sim.setBatchSize(1)
        sim.initialize()
        for ix in range(36):
            if ix == 0:
                sim.newEpisode([scan], [viewpoint], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            if state.viewIndex == viewidx:
                image = np.array(state.rgb, copy=True)
                Image.fromarray(image[:, :, ::-1]).save(out_path)
                return out_path
        raise RuntimeError(f"Could not render viewidx {viewidx} for {scan}/{viewpoint}")

    def eval_metrics_update_all_preds(self, preds, logger, name):
        """
        Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]
        :param preds:
        :param logger:
        :param name:
        :return:
        """
        logger.info('eval %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for i, item in enumerate(preds):
            instr_id = item['instr_id']
            traj = item['trajectory']

            if instr_id not in self.gt_trajs.keys():
                print("instr_id {} not in self.gt_trajs".format(instr_id))
                raise NotImplementedError

            if name == "R2R":
                scan, gt_traj = self.gt_trajs[instr_id]
                traj_scores = self.eval_dis_item(scan, traj, gt_traj)
            else:
                raise NotImplementedError

            # if self.args.add_action_prediction_in_navigational_reasoning:
            #     traj_scores['consistency_rate'] = float(list(preds[i]['cot_decision_consistency'].values()).count(True)/len(list(preds[i]['cot_decision_consistency'].values())))

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

            preds[i]['nav_error'] = traj_scores['nav_error']
            preds[i]['oracle_error'] = traj_scores['oracle_error']
            preds[i]['success'] = traj_scores['success']
            if 'consistency_rate' in traj_scores:
                preds[i]['consistency_rate'] = traj_scores['consistency_rate']
            if preds[i]['success'] != 1.:
                preds[i]['failure_reason'] = 'stop' if traj_scores['oracle_success'] == 1. else 'exploration'



        if name in ['R2R']:
            avg_metrics = {
                'action_steps': np.mean(metrics['action_steps']),
                'steps': np.mean(metrics['trajectory_steps']),
                'lengths': np.mean(metrics['trajectory_lengths']),
                'nav_error': np.mean(metrics['nav_error']),
                'oracle_error': np.mean(metrics['oracle_error']),
                'sr': np.mean(metrics['success']) * 100,
                'oracle_sr': np.mean(metrics['oracle_success']) * 100,
                'spl': np.mean(metrics['spl']) * 100,
                'consistency_rate': np.mean(metrics['consistency_rate']) * 100 if "consistency_rate" in metrics else -1
            }
        else:
            raise NotImplementedError
        #return preds, avg_metrics, metrics
        return avg_metrics, metrics

    def eval_dis_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self.get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        return scores

    def save_json(self, results, path, item_metrics=None):
        if item_metrics is not None:
            for k in item_metrics:
                for item, v in zip(results, item_metrics[k]):
                    item[k] = v

        for item in results:
            item['instr_id'] = "_".join(item['instr_id'].split("_")[1:])
            item['trajectory'] = [[y, 0, 0] for x in item['trajectory'] for y in x]

        with open(path, 'w') as fout:
            json.dump(results, fout)
