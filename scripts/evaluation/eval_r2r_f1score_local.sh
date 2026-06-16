# Local machine phase after copying the output_dir:
export OPENAI_API_KEY='add your api key here'
python scripts/evaluation/compute_action_reasoning_f1_from_facts.py \
   --facts_file build/eval/20260615_F1score/epoch_15/R2R_val_unseen_action_reasoning_facts.json \
   --output_dir build/f1_local/val_unseen_epoch_15 \
   --image_root build/eval/20260615_F1score/epoch_15/action_reasoning_judge_images \
   --model gpt-4o \
   --progress_every 1
