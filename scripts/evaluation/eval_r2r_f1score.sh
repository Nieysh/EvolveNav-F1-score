# Single-view images should be laid out as:
#   $ACTION_REASONING_IMAGE_DIR/{scan}/{viewpoint}/{viewidx}.jpg
ACTION_REASONING_IMAGE_DIR=${ACTION_REASONING_IMAGE_DIR:-}

CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=5 --master_port 16931 train_wlora_new.py \
    --stage multi --mode test --cfg_file configs/r2r.yaml \
    --data_dir data --pretrained_model_name_or_path data/Vicuna-7B-v1.1 --precision amp_bf16 \
    --val_batch_size 2 \
    --test_datasets R2R \
    --resume_from_checkpoint "output/r2r-lora-once_forward_cot_navigation-resumebest-cot_summarization-max5ldm-cotv4-cotoutputassupervision-selfselectlossweight0.2-alternateposttraining-lora/epoch_15.pt" \
    --output_dir build/eval/20260615_F1score/r2r-lora-once_forward_cot_navigation-resumebest-cot_summarization-max5ldm-cotv4-cotoutputassupervision-selfselectlossweight0.2-alternateposttraining-lora_epoch_15 \
    --save_latest_states --validation_split val_unseen --save_pred_results \
    --enable_lora \
    --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --lora_target_modules "all_linear" \
    --lora_bias "none" \
    --enable_navigation_cot --cot_summarization --cot_v4 --enable_og --enable_summarize --enable_fgr2r \
    --remove_summarization \
    --enable_action_reasoning_f1 \
    --action_reasoning_image_dir "$ACTION_REASONING_IMAGE_DIR" \
    --eval_progress_every 1 \
    --action_reasoning_progress_every 1 \
    --action_reasoning_print_examples 3 \
    --action_reasoning_eval_phase infer
