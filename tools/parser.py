import argparse
import random
import numpy as np
import torch
import os
import datetime
import yaml
from easydict import EasyDict
from .distributed import world_info_from_env, init_distributed_device
from .common_utils import create_logger, log_config_to_file
from pathlib import Path


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def read_args():
    parser = argparse.ArgumentParser()

    ### newly added: spatial reasoning sub tasks
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb mode")
    parser.add_argument("--wandb_project", type=str, default="NaviLLM", help="Wandb project name")
    parser.add_argument("--all_cand_input", action="store_true", help="")
    parser.add_argument("--enable_navigation_cot", action="store_true", help="")
    parser.add_argument("--test_with_cot_gt", action="store_true", help="")
    parser.add_argument("--base_model", type=str, default=None, help="")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--add_cand_landmark", action="store_true", default=False, help="")
    parser.add_argument("--check_cot_input_gt", action="store_true", default=False, help="")
    parser.add_argument("--self_improving_cot", action="store_true", default=False, help="")
    parser.add_argument("--only_IL", action="store_true", default=False, help="")
    parser.add_argument("--cot_summarization", action="store_true", default=False, help="")
    parser.add_argument("--visualize", action="store_true", default=False, help="")
    parser.add_argument("--cot_v4", action="store_true", default=False, help="")
    parser.add_argument("--enable_self_refine", action="store_true", default=False, help="")
    parser.add_argument("--enable_self_select", action="store_true", default=False, help="")
    parser.add_argument("--cot_v4_only_direction", action="store_true", default=False, help="")
    parser.add_argument('--cal_lmloss_prob', type=float, default=0.5, help="")
    parser.add_argument('--cal_lmloss_prob_nor2r', type=float, default=0.5, help="")
    parser.add_argument("--add_lmloss_with_prob", action="store_true", default=False, help="")
    parser.add_argument("--remove_summarization", action="store_true", default=False, help="")
    parser.add_argument('--enable_RL_A2C', action="store_true", default=False, help="")
    parser.add_argument('--normalize_loss', type=str, default='batch', help="")
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01, help="")
    parser.add_argument('--gamma', type=float, default=0.9, help="")
    parser.add_argument('--resume_from_critic_checkpoint', type=str, default=None)
    parser.add_argument('--landmark_not_merge_in_gt', action="store_true", default=False, help="")
    parser.add_argument('--step_wise_a2c', action="store_true", default=False, help="")
    parser.add_argument('--enable_deepspeed', action="store_true", default=False, help="")
    parser.add_argument('--enable_RL_GRPO', action="store_true", default=False, help="")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--action_first_in_gt', action="store_true", default=False, help="")
    parser.add_argument('--random_target_vp_in_cot_gt', action="store_true", default=False, help="This only changes cot gt, no influence on navigation target")
    parser.add_argument('--multiple_sample_cot', action="store_true", default=False, help="")
    parser.add_argument('--cot_first_in_gt', action="store_true", default=False, help="")
    parser.add_argument('--greedy_first_eval', action="store_true", default=False, help="")
    parser.add_argument('--self_improve_wo_orisft', action="store_true", default=False, help="")
    parser.add_argument('--cot_sample_return_sequences', type=int, default=5, help="")
    parser.add_argument('--cot_sample_temperature', type=float, default=0.1, help="")
    parser.add_argument('--alternate_IL_RL', action="store_true", default=False, help="")
    parser.add_argument('--alternate_dagger_RL', action="store_true", default=False, help="")
    parser.add_argument('--add_efficiency_reward', action="store_true", default=False, help="")
    parser.add_argument('--cot_output_as_supervision', action="store_true", default=False, help="")
    parser.add_argument('--replace_cot_gt_with_prob', action="store_true", default=False, help="")
    parser.add_argument('--replace_cot_gt_prob', type=float, default=0.5)
    parser.add_argument('--mlm', action="store_true", default=False, help="")
    parser.add_argument('--land_token_region_length', type=int, default=30)
    parser.add_argument('--add_selfrefine_loss_with_prob', action="store_true", default=False, help="")
    parser.add_argument('--add_selfselect_loss_with_prob', action="store_true", default=False, help="")
    parser.add_argument('--self_select_loss_weight', type=float, default=1., help="")
    parser.add_argument('--self_refine_loss_weight', type=float, default=1., help="")
    parser.add_argument('--remove_qa_prompt_v2', action="store_true", default=False, help="")
    parser.add_argument("--cot_v4_only_landmark", action="store_true", default=False, help="")
    parser.add_argument('--land_num', type=int, default=5)
    parser.add_argument("--sft_warmup", action="store_true", default=False, help="")
    parser.add_argument('--sft_warmup_epoch_num', type=int, default=10)
    parser.add_argument("--alternate_post_training", action="store_true", default=False, help="")
    parser.add_argument("--visualize_cot", action="store_true", default=False, help="")

    ### newly added: lora config
    parser.add_argument('--enable_lora', action='store_true', default=False, help="whether to use lora")
    parser.add_argument('--lora_r', type=int, default=None, help="lora rank")
    parser.add_argument('--lora_alpha', type=int, default=None, help="lora alpha")
    parser.add_argument('--lora_dropout', type=float, default=None, help="lora dropout")
    parser.add_argument('--lora_bias', type=str, default=None, help="lora bias")
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="lora target")
    parser.add_argument('--lora_weight_path', type=str, default=None, help="lora weight")

    parser.add_argument('--data_dir', type=str, default='data', help="dataset root path")
    parser.add_argument('--cfg_file', type=str, default=None, help='dataset configs', required=True)
    parser.add_argument('--pretrained_model_name_or_path', default=None, type=str, required=True, help="path to tokenizer")

    # local fusion
    parser.add_argument('--off_batch_task', action='store_true', default=False, help="whether all process is training same task")
    parser.add_argument('--debug', action="store_true", help="debug mode")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to ckpt to resume from")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--feat_dropout", type=float, default=0.4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument("--workers", type=int, default=0)

    # distributed training args
    parser.add_argument('--world_size', type=int, default=0, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # Save checkpoints
    parser.add_argument('--output_dir', type=str, default=None, required=True, help="output logs and ckpts")
    parser.add_argument("--max_saved_checkpoints", type=int, default=0)
    parser.add_argument("--save_ckpt_per_epochs", type=int, default=10)
    parser.add_argument("--save_pred_results", action="store_true")
    parser.add_argument("--save_latest_states", action='store_true')
    parser.add_argument("--save_detail_results", action="store_true")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument("--stage", type=str, required=True, choices=["pretrain", "multi"])
    parser.add_argument('--ignoreid', default=-100, type=int, help="criterion: ignore label")
    parser.add_argument('--enable_og', action='store_true', default=False, help="object grounding task")
    parser.add_argument("--enable_summarize", action="store_true", help="perform EQA or generate instructions")
    parser.add_argument("--enable_fgr2r", action="store_true", help="perform fgr2r for R2R")
    parser.add_argument("--gen_loss_coef", type=float, default=1.)
    parser.add_argument("--obj_loss_coef", type=float, default=1.)
    parser.add_argument("--teacher_forcing_coef", type=float, default=1.)
    parser.add_argument("--cotsum_loss_coef", type=float, default=1.)
    parser.add_argument("--fuse_obj", action="store_true", help="whether fuse object features for REVERIE and SOON")

    # datasets
    parser.add_argument("--multi_endpoints", type=int, default=1)
    parser.add_argument("--path_type", type=str, default="trusted_path", choices=["planner_path", "trusted_path"])

    # evaluation
    parser.add_argument('--test_datasets', type=str, default=None, nargs='+')
    parser.add_argument('--validation_split', type=str, default="val_unseen",
                        help="validation split: val_seen, val_unseen, test")
    parser.add_argument("--do_sample", action="store_true", help="do_sample in evaluation")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--enable_action_reasoning_f1", action="store_true",
                        help="Evaluate action-reasoning-F1 from generated CoT and action correctness.")
    parser.add_argument("--action_reasoning_judge_mode", type=str, default="text",
                        choices=["text", "vlm"],
                        help="Judge reasoning landmarks with local text fallback or an external VLM.")
    parser.add_argument("--action_reasoning_eval_phase", type=str, default="both",
                        choices=["infer", "both"],
                        help="both: judge during evaluation. infer: only export action-reasoning facts/images for offline scoring.")
    parser.add_argument("--action_reasoning_facts_file", type=str, default=None,
                        help="Optional output path for exported action-reasoning facts in infer phase.")
    parser.add_argument("--action_reasoning_vlm_model", type=str, default="gpt-4o-mini",
                        help="OpenAI-compatible VLM model name for action-reasoning judging.")
    parser.add_argument("--action_reasoning_vlm_api_key_env", type=str, default="OPENAI_API_KEY",
                        help="Environment variable containing the VLM API key.")
    parser.add_argument("--action_reasoning_vlm_base_url", type=str,
                        default="https://api.openai.com/v1/chat/completions",
                        help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--action_reasoning_image_dir", type=str, default=None,
                        help="Optional root directory containing rendered MP3D action-view images.")
    parser.add_argument("--action_reasoning_image_pattern", type=str, default=None,
                        help="Optional image path pattern relative to image_dir, using {scan}, {viewpoint}, {viewidx}.")
    parser.add_argument("--action_reasoning_scan_dir", type=str, default=None,
                        help="Optional MP3D scan data dir for rendering action-view images with MatterSim.")
    parser.add_argument("--action_reasoning_image_cache_dir", type=str, default=None,
                        help="Directory for MatterSim-rendered judge images. Defaults to output_dir/action_reasoning_images.")
    parser.add_argument("--action_reasoning_print_examples", type=int, default=3,
                        help="Number of action-reasoning intermediate examples to print/log.")
    parser.add_argument("--action_reasoning_progress_every", type=int, default=10,
                        help="Print/log action-reasoning judge progress every N judged steps.")
    parser.add_argument("--eval_progress_every", type=int, default=10,
                        help="Print model-evaluation progress every N validation batches.")
    parser.add_argument("--eval_episode_limit", type=int, default=None,
                        help="Limit validation/test samples for quick evaluation checks.")

    # others
    parser.add_argument(
        "--max_datapoints",
        default=None,
        type=int,
        help="The number of datapoints used for debug."
    )

    args = parser.parse_args()

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.multiple_sample_cot:
        random_seed()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ###################### configurations #########################
    # single-gpu or multi-gpu
    device_id = init_distributed_device(args)
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))

    args.data_dir = Path(args.data_dir).resolve()

    # off-line image features from Matterport3D
    args.image_feat_size = global_cfg.Feature.image_feat_size
    args.obj_feat_size = global_cfg.Feature.obj_feat_size

    ############# Configurations ###############
    args.angle_feat_size = global_cfg.Feature.angle_feat_size
    args.enc_full_graph = global_cfg.Model.enc_full_graph
    args.expert_policy = global_cfg.Model.expert_policy
    args.num_pano_layers = global_cfg.Model.num_pano_layers

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = Path(args.output_dir) / 'log.txt'

    logger = create_logger(log_file, rank=args.rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(global_cfg, logger=logger)

    print(" + rank: {}, + device_id: {}".format(args.local_rank, device_id))
    print(f"Start running training on rank {args.rank}.")

    if os.path.exists(os.path.join(args.output_dir, "latest_states.pt")):
        state_path = os.path.join(args.output_dir, "latest_states.pt")
        logger.info("Resume checkponit from {}".format(state_path))
        args.resume_from_checkpoint = state_path

    return args, global_cfg, logger, device_id
