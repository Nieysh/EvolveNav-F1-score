import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
os.environ['OPENAI_API_KEY'] = 'add you api key here'

DEFAULT_FALSE_TAG = "vlm_judge_default_false_after_retries"

def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", str(text), re.S)
        if match:
            return json.loads(match.group(0))
        raise


def resolve_image_path(fact, facts_dir, image_root=None):
    relpath = fact.get("judge_image_relpath")
    if relpath:
        path = facts_dir / relpath
        if path.exists():
            return path
    judge_path = fact.get("judge_image_path")
    if judge_path and Path(judge_path).exists():
        return Path(judge_path)
    source_path = fact.get("source_image_path")
    if image_root and source_path:
        source = Path(source_path)
        for suffix_parts in (
            source.parts[-4:],
            source.parts[-3:],
            source.parts[-2:],
            source.parts[-1:],
        ):
            candidate = Path(image_root).joinpath(*suffix_parts)
            if candidate.exists():
                return candidate
    return None


def call_vlm(fact, image_path, args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    prompt = (
        "You are judging one navigation reasoning step. The image is the ground-truth action view. "
        "Decide whether the landmarks/objects/places mentioned in the model reasoning are visible "
        "in the image, allowing reasonable synonyms and visually equivalent descriptions. "
        "Do not require exact wording. If the reasoning mentions no visual landmark/object/place, answer false. "
        "Return only JSON with keys: landmarks_visible (boolean), visible_evidence (list of strings), "
        "missing_or_uncertain (list of strings), explanation (short string).\n\n"
        f"Model reasoning: {fact.get('reasoning', '')}\n"
        f"Parsed landmark phrases: {fact.get('predicted_landmarks', [])}\n"
        f"GT next viewpoint id: {fact.get('gt_vpid')}\n"
        f"Instruction id: {fact.get('instr_id', '')}, step: {fact.get('step', '')}"
    )
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        args.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(1, args.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            return safe_json_loads(content)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if attempt == args.retries:
                raise RuntimeError(f"VLM judge HTTP error {exc.code}: {body}") from exc
        except Exception:
            if attempt == args.retries:
                raise
        time.sleep(args.retry_sleep * attempt)


def classify(action_correct, reasoning_correct):
    if action_correct and reasoning_correct:
        return "TP"
    if (not action_correct) and reasoning_correct:
        return "FP"
    if action_correct and (not reasoning_correct):
        return "FN"
    return "TN"


def fact_key(fact):
    return json.dumps(
        {
            "instr_id": fact.get("instr_id", ""),
            "step": fact.get("step", ""),
            "gt_vpid": fact.get("gt_vpid", ""),
            "judge_image_relpath": fact.get("judge_image_relpath", ""),
            "judge_image_path": fact.get("judge_image_path", ""),
            "source_image_path": fact.get("source_image_path", ""),
        },
        sort_keys=True,
    )


def compute_metrics(counts):
    precision_den = counts["TP"] + counts["FP"]
    recall_den = counts["TP"] + counts["FN"]
    precision = counts["TP"] / precision_den if precision_den else 0.0
    recall = counts["TP"] / recall_den if recall_den else 0.0
    f1_den = precision + recall
    f1 = 2 * precision * recall / f1_den if f1_den else 0.0
    return {
        "action_reasoning_precision": precision * 100,
        "action_reasoning_recall": recall * 100,
        "action_reasoning_f1": f1 * 100,
        "action_reasoning_tp": counts["TP"],
        "action_reasoning_fp": counts["FP"],
        "action_reasoning_fn": counts["FN"],
        "action_reasoning_tn": counts["TN"],
    }


def build_output(facts_file, args, facts, judged, counts):
    return {
        "facts_file": str(facts_file),
        "model": args.model,
        "num_steps": len(facts),
        "metrics": compute_metrics(counts),
        "default_false_tag": DEFAULT_FALSE_TAG,
        "default_false_steps": sum(
            1 for item in judged if item.get("vlm_judge_status") == DEFAULT_FALSE_TAG
        ),
        "judged_steps": judged,
    }


def save_results(path, output, replace_retries=10, retry_sleep=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for attempt in range(1, replace_retries + 1):
        try:
            os.replace(tmp_path, path)
            return path
        except PermissionError as exc:
            if attempt == replace_retries:
                fallback_path = path.with_name(
                    f"{path.stem}.save_failed_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
                )
                try:
                    os.replace(tmp_path, fallback_path)
                    print(
                        f"[ActionReasoningF1] warning: could not replace locked file {path}: {exc}. "
                        f"Saved fallback {fallback_path}"
                    )
                    return fallback_path
                except Exception:
                    raise exc
            time.sleep(retry_sleep * attempt)


def load_resume_results(path):
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ActionReasoningF1] warning: could not load resume file {path}: {exc}")
        return {}

    results = {}
    for item in payload.get("judged_steps", []):
        results[fact_key(item)] = item
    return results


def make_default_false_judge(exc, args):
    return {
        "landmarks_visible": False,
        "default_false": True,
        "tag": DEFAULT_FALSE_TAG,
        "attempts": args.retries,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts_file", default='build/eval/20260615_F1score/epoch_15/R2R_val_unseen_action_reasoning_facts.json')
    parser.add_argument("--output_dir", default='build/f1_local/val_unseen_epoch_15')
    parser.add_argument("--image_root", default='build/eval/20260615_F1score/epoch_15/action_reasoning_judge_images', help="Optional local image root if copied judge images are unavailable.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--intermediate_save_every", type=int, default=1)
    parser.add_argument("--resume_intermediate", dest="resume_intermediate", action="store_true", default=True)
    parser.add_argument("--no_resume_intermediate", dest="resume_intermediate", action="store_false")
    args = parser.parse_args()

    facts_file = Path(args.facts_file).resolve()
    facts_dir = facts_file.parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else facts_dir / "local_action_reasoning_f1"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(facts_file.read_text(encoding="utf-8"))
    facts = payload.get("facts", payload if isinstance(payload, list) else [])
    if args.limit is not None:
        facts = facts[:args.limit]

    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    judged = []
    progress_every = max(1, args.progress_every)
    intermediate_save_every = max(1, args.intermediate_save_every)
    result_path = output_dir / "action_reasoning_f1_results.json"
    intermediate_path = output_dir / "action_reasoning_f1_results.intermediate.json"
    resume_results = {}
    if args.resume_intermediate:
        resume_results = load_resume_results(intermediate_path)
        if not resume_results:
            resume_results = load_resume_results(result_path)
    print(f"[ActionReasoningF1] start facts={len(facts)} model={args.model}")
    if resume_results:
        print(f"[ActionReasoningF1] loaded resume results={len(resume_results)}")

    for idx, fact in enumerate(facts, start=1):
        existing = resume_results.get(fact_key(fact))
        if existing and existing.get("vlm_judge_status") != DEFAULT_FALSE_TAG:
            label = existing.get("label")
            if label in counts:
                counts[label] += 1
                judged.append(existing)
                if idx == 1 or idx == len(facts) or idx % progress_every == 0:
                    print(
                        f"[ActionReasoningF1] progress {idx}/{len(facts)} "
                        f"instr_id={fact.get('instr_id', '')} step={fact.get('step', '')} "
                        f"label={label} reused=True"
                    )
                continue
        elif existing and existing.get("vlm_judge_status") == DEFAULT_FALSE_TAG:
            print(
                f"[ActionReasoningF1] retrying default-false VLM judge "
                f"instr_id={fact.get('instr_id', '')} step={fact.get('step', '')} "
                f"tag={DEFAULT_FALSE_TAG}"
            )

        action_correct = bool(fact.get("action_correct", False))
        direction_correct = bool(fact.get("direction_correct", False))
        judge = None
        vlm_judge_status = "ok"
        image_path = resolve_image_path(fact, facts_dir, args.image_root)

        if fact.get("judge_skipped") == "stop_action":
            landmark_correct = True
            judge = {"skipped": "stop_action"}
            vlm_judge_status = "skipped_stop_action"
        elif direction_correct:
            landmark_correct = False
            judge = {"skipped": "direction_correct"}
            vlm_judge_status = "skipped_direction_correct"
        else:
            if image_path is None:
                landmark_correct = False
                judge = {"error": "missing_image"}
                vlm_judge_status = "missing_image"
            else:
                try:
                    judge = call_vlm(fact, image_path, args)
                    landmark_correct = bool(judge.get("landmarks_visible", False))
                except Exception as exc:
                    landmark_correct = False
                    judge = make_default_false_judge(exc, args)
                    vlm_judge_status = DEFAULT_FALSE_TAG
                    print(
                        f"[ActionReasoningF1] warning: VLM judge failed after {args.retries} attempts; "
                        f"defaulting false with tag={DEFAULT_FALSE_TAG} "
                        f"instr_id={fact.get('instr_id', '')} step={fact.get('step', '')}: {exc}"
                    )

        if fact.get("judge_skipped") == "stop_action":
            reasoning_correct = direction_correct
        else:
            reasoning_correct = direction_correct or landmark_correct
        label = classify(action_correct, reasoning_correct)
        counts[label] += 1
        result = {
            **fact,
            "local_image_path": str(image_path) if image_path else None,
            "landmark_correct": landmark_correct,
            "reasoning_correct": reasoning_correct,
            "label": label,
            "vlm_judge_status": vlm_judge_status,
            "vlm_judge_raw": judge,
        }
        judged.append(result)
        if idx == len(facts) or idx % intermediate_save_every == 0:
            save_results(
                intermediate_path,
                build_output(facts_file, args, facts, judged, counts),
            )

        if idx == 1 or idx == len(facts) or idx % progress_every == 0:
            print(
                f"[ActionReasoningF1] progress {idx}/{len(facts)} "
                f"instr_id={fact.get('instr_id', '')} step={fact.get('step', '')} "
                f"label={label} action_correct={action_correct} reasoning_correct={reasoning_correct}"
            )

    output = build_output(facts_file, args, facts, judged, counts)
    metrics = output["metrics"]
    save_results(result_path, output)
    save_results(intermediate_path, output)
    print(
        "[ActionReasoningF1] "
        f"TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} TN={counts['TN']} "
        f"P={metrics['action_reasoning_precision']:.2f} "
        f"R={metrics['action_reasoning_recall']:.2f} "
        f"F1={metrics['action_reasoning_f1']:.2f}"
    )
    print(
        f"[ActionReasoningF1] default_false_tag={DEFAULT_FALSE_TAG} "
        f"default_false_steps={output['default_false_steps']}"
    )
    print(f"[ActionReasoningF1] saved {result_path}")


if __name__ == "__main__":
    main()
