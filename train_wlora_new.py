import os
import shutil
import json
import torch
import random
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import torch.nn as nn
from tools.common_utils import all_gather
from tools.parser import read_args, random_seed
from tasks.loaders import create_dataloaders
from tasks.feature_db import create_feature_db, create_object_feature_db
from models.nav_model import NavModel
from tools.optims_new_lora import dist_models, save_checkpoint
from tools.trie import Trie
from torch.distributed.elastic.multiprocessing.errors import record
# from deepspeed import zero
# from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value

# def maybe_zero_3(param):
#     if hasattr(param, "ds_id"):
#         assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param
#
# # Borrowed from peft.utils.get_peft_model_state_dict
# def get_peft_state_maybe_zero_3(named_params, bias):
#     if bias == "none":
#         to_return = {k: t for k, t in named_params if "lora_" in k}
#     elif bias == "all":
#         to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
#     elif bias == "lora_only":
#         to_return = {}
#         maybe_lora_bias = {}
#         lora_bias_names = set()
#         for k, t in named_params:
#             if "lora_" in k:
#                 to_return[k] = t
#                 bias_name = k.split("lora_")[0] + "bias"
#                 lora_bias_names.add(bias_name)
#             elif "bias" in k:
#                 maybe_lora_bias[k] = t
#         for k, t in maybe_lora_bias:
#             if bias_name in lora_bias_names:
#                 to_return[bias_name] = t
#     else:
#         raise NotImplementedError
#     to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
#     return to_return
#
#
# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return


class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        if self.num == 0:
            return 0
        return self.total / self.num


def train_one_epoch(
        args,
        global_cfg,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        dataloaders,
        agents,
        epoch,
        logger,
        stage='multi'
):

    model.train()
    entropy_metric = Metrics()
    loss_metric = Metrics()
    instr_pred_metric = Metrics()
    cnt_loss_metric = Metrics()
    ml_loss_metric = Metrics()
    if args.enable_self_select:
        self_select_loss_metric = Metrics()

    num_batches_per_epoch = dataloaders.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        range(dataloaders.num_batches),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    
    dataset_cfg = global_cfg.Pretrain if stage=='pretrain' else global_cfg.Multi
    loss_stats = {k: Metrics() for k in dataset_cfg.SOURCE}

    for step, (name, batch) in enumerate(dataloaders):
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        # perform embodied tasks
        # the actual batch_size equals to args.batch_size * world_size * (args.gradient_accumulation_step)
        dataset = dataloaders.loader.get_dataset(name)
        agent = agents.get(name)
        if args.enable_self_select:
            loss, self_select_loss = agent.train(
                name,
                batch,
                args,
                global_cfg,
                model=model,
                criterion=criterion,
                dataset=dataset,
                step=step,
                entropy_metric=entropy_metric,
                instr_pred_metric=instr_pred_metric,
                cnt_loss_metric=cnt_loss_metric,
                ml_loss_metric=ml_loss_metric,
                epoch=epoch,
            )
            self_select_loss_metric.accumulate(self_select_loss.item())
        else:
            loss = agent.train(
                name,
                batch,
                args,
                global_cfg,
                model=model,
                criterion=criterion,
                dataset=dataset,
                step=step,
                entropy_metric=entropy_metric,
                instr_pred_metric=instr_pred_metric,
                cnt_loss_metric=cnt_loss_metric,
                ml_loss_metric=ml_loss_metric,
                epoch=epoch,
            )
        loss_metric.accumulate(loss.item())
        loss_stats[name].accumulate(loss.item())
        torch.cuda.empty_cache()

        if (step+1) % args.gradient_accumulation_step==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.)
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        if args.rank == 0:
            verbose_dict = dict(
                step=step,
                name=name,
                # index=batch['sample_idx'],
                loss=loss_metric.average,
                entropy=entropy_metric.average,
                instr_pred_metric=instr_pred_metric.average,
                lr=lr_scheduler.get_last_lr()[0],
            )
            for k in dataset_cfg.SOURCE:
                verbose_dict[k] = loss_stats[k].average
            pbar.set_postfix(verbose_dict)
            pbar.update()

            if args.enable_wandb:
                wandb.log({"epoch": epoch,
                        f"{name}_loss": loss_stats[name].average,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "entropy_metric": entropy_metric.average,
                        "instr_pred_metric": instr_pred_metric.average,
                        "cnt_loss": cnt_loss_metric.average,  # 新增
                        "ml_loss": ml_loss_metric.average,     # 新增
                        "loss": loss_metric.average
                }, step=epoch*args.num_steps_per_epoch+step)
            # wandb.log({"epoch": epoch,
            #            f"{name}_loss": loss_stats[name].average,
            #            "lr": lr_scheduler.get_last_lr()[0],
            #            "entropy_metric": entropy_metric.average,
            #            "instr_pred_metric": instr_pred_metric.average,
            #            "loss": loss_metric.average}, step=epoch*args.num_steps_per_epoch+step)

        if step == num_batches_per_epoch-1:
            logger.info("***** train [{}] epoch *****".format(epoch))
            train_stat_str = 'Loss: %.2f\n' % loss_metric.average
            if args.enable_self_select:
                train_stat_str += 'self select Loss: %.2f\n' % self_select_loss_metric.average
            train_stat_str += "Instr_pred: %.2f\n" % instr_pred_metric.average
            for task in dataset_cfg.SOURCE:
                train_stat_str += "%s: %.2f\n" % (task, loss_stats[task].average)
                if args.rank == 0 and args.enable_wandb:
                    wandb.log({"epoch": epoch,
                            "task": task,
                            "loss_accum": loss_stats[task].average}, step=epoch*args.num_steps_per_epoch+step)
            logger.info(train_stat_str)
            break

def train_one_epoch_RL(        args,
        global_cfg,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        dataloaders,
        agents,
        epoch,
        logger,
        stage='multi',
        critic=None,
        critic_optimizer=None,
        critic_lr_scheduler=None):
    model.train()
    critic.train() # newly added
    entropy_metric = Metrics()
    loss_metric = Metrics()
    instr_pred_metric = Metrics()

    num_batches_per_epoch = dataloaders.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        range(dataloaders.num_batches),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )

    dataset_cfg = global_cfg.Pretrain if stage == 'pretrain' else global_cfg.Multi
    loss_stats = {k: Metrics() for k in dataset_cfg.SOURCE}

    for step, (name, batch) in enumerate(dataloaders):
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        # perform embodied tasks
        # the actual batch_size equals to args.batch_size * world_size * (args.gradient_accumulation_step)
        dataset = dataloaders.loader.get_dataset(name)
        agent = agents.get(name)
        loss = agent.train(
            name,
            batch,
            args,
            global_cfg,
            model=model,
            criterion=criterion,
            dataset=dataset,
            step=step,
            entropy_metric=entropy_metric,
            instr_pred_metric=instr_pred_metric,
            critic=critic,
        )
        loss_metric.accumulate(loss.item())
        loss_stats[name].accumulate(loss.item())

        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.)
            optimizer.step()
            # newly added
            critic_optimizer.step()

            optimizer.zero_grad()
            # newly added
            critic_optimizer.zero_grad()

        lr_scheduler.step()
        critic_lr_scheduler.step() # newly added

        if args.rank == 0:
            verbose_dict = dict(
                step=step,
                name=name,
                # index=batch['sample_idx'],
                loss=loss_metric.average,
                entropy=entropy_metric.average,
                instr_pred_metric=instr_pred_metric.average,
                lr=lr_scheduler.get_last_lr()[0],
                critic_lr=critic_lr_scheduler.get_last_lr()[0]
            )
            for k in dataset_cfg.SOURCE:
                verbose_dict[k] = loss_stats[k].average
            pbar.set_postfix(verbose_dict)
            pbar.update()

        if step == num_batches_per_epoch - 1:
            logger.info("***** train [{}] epoch *****".format(epoch))
            train_stat_str = 'Loss: %.2f\n' % loss_metric.average
            train_stat_str += "Instr_pred: %.2f\n" % instr_pred_metric.average
            for task in dataset_cfg.SOURCE:
                train_stat_str += "%s: %.2f\n" % (task, loss_stats[task].average)
            logger.info(train_stat_str)
            break

@torch.no_grad()
def val_one_epoch(
        args,
        global_cfg,
        model,
        optimizer,
        criterion,
        dataloaders,
        agents,
        epoch,
        logger,
) -> Dict[str, Dict[str, float]]:

    model.eval()
    entropy_metric = Metrics()

    loss_str = "\n[Eval] {} epoch {}\n".format(args.validation_split, epoch)
    task_results = {}
    for name, loader in dataloaders.items():
        item_metrics = None
        logger.info("***** validate {} split on {} task *****".format(args.validation_split, name))
        dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        preds = agent.validate(
            name,
            args,
            global_cfg,
            model,
            loader,
            entropy_metric=entropy_metric
        )

        all_preds = all_gather(preds)
        all_preds = merge_dist_results(all_preds)

        action_reasoning_infer_phase = (
            args.enable_action_reasoning_f1
            and getattr(args, "action_reasoning_eval_phase", "both") == "infer"
        )
        if args.rank == 0 and action_reasoning_infer_phase:
            facts_path = args.action_reasoning_facts_file
            if facts_path is None:
                facts_path = os.path.join(
                    args.output_dir,
                    f"{name}_{args.validation_split}_action_reasoning_facts.json",
                )
            dataset.export_action_reasoning_facts(all_preds, facts_path, logger=logger, name=name)
            logger.info(
                "[ActionReasoningF1] infer phase: skipped VLM/API scoring. "
                "Use scripts/evaluation/compute_action_reasoning_f1_from_facts.py locally with %s",
                facts_path,
            )

        action_reasoning_score_phase = (
            args.enable_action_reasoning_f1
            and getattr(args, "action_reasoning_eval_phase", "both") == "both"
        )
        should_eval = not args.validation_split.startswith('test') or action_reasoning_score_phase
        if args.rank == 0 and should_eval:
            #if args.enable_navigation_cot:
            #    all_preds, score_summary, item_metrics = dataset.eval_metrics_update_all_preds(all_preds, logger=logger, name=name)
            #else:
            score_summary, item_metrics = dataset.eval_metrics(all_preds, logger=logger, name=name)

            task_results[name] = score_summary
            loss_str += "\n [Eval] dataset=[{}] \n".format(name)
            for metric, val in score_summary.items():
                if metric == 'sr':
                    loss_str += '\n[Eval] ||| %s: %.2f' % (metric, val)
                else:
                    loss_str += ', %s: %.2f' % (metric, val)

                if args.enable_wandb:
                    wandb.log({f"{name}-{metric}": val}, step=(epoch + 1) * args.num_steps_per_epoch - 1)

            summary_path = os.path.join(args.output_dir, f"{name}_{args.validation_split}_metrics.json")
            with open(summary_path, "w") as fout:
                json.dump(json_safe({
                    "dataset": name,
                    "split": args.validation_split,
                    "epoch": epoch,
                    "metrics": score_summary,
                    "num_predictions": len(all_preds),
                }), fout, indent=2)
            logger.info("Saved evaluation summary to %s", summary_path)

        if args.rank== 0 and args.save_pred_results:
            dataset.save_json(
                all_preds, 
                os.path.join(args.output_dir, f"{name}_{args.validation_split}.json"),
                item_metrics=item_metrics if args.save_detail_results else None
            )


    logger.info(loss_str)
    
    return task_results



def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)

    # DistributedSampler pads evaluation shards when the dataset size is not
    # divisible by world size. Keep the first prediction for each instruction
    # so rank-0 metrics/export are not skewed by duplicated samples.
    deduped = []
    seen = set()
    for item in outs:
        instr_id = item.get("instr_id") if isinstance(item, dict) else None
        if instr_id is None:
            deduped.append(item)
            continue
        if instr_id in seen:
            continue
        seen.add(instr_id)
        deduped.append(item)
    return deduped

def calc_overall_score(results, cfg):
    score = 0.
    for task in results:
        if task not in cfg.Multi.SOURCE:
            continue
        if task == 'R2R':
            score += results[task]['spl'] / 60
        elif task == 'REVERIE':
            score += results[task]['spl'] / 36.63
        elif task == 'CVDN':
            score += results[task]['dist_to_end_reduction']
        elif task == 'SOON':
            score += results[task]['spl'] / 26.58
        elif task == 'EQA':
            pass
        elif task == "ScanQA":
            pass
        else:
            raise NotImplementedError(f"The method for calculating the score of {task} is not Implemented.")

    return score

# def calc_overall_score(results, cfg):
#     score = 0.
#     for task in results:
#         if task not in cfg.Multi.SOURCE:
#             continue
#         if task == 'R2R':
#             score += results[task]['spl'] / 60
#         elif task == 'REVERIE':
#             score += results[task]['spl'] / 36.63
#         elif task == 'CVDN':
#             pass
#         elif task == 'SOON':
#             score += results[task]['spl'] / 26.58
#         elif task == 'EQA':
#             pass
#         elif task == "ScanQA":
#             pass
#         elif task == "RoomTour":
#             pass
#         else:
#             raise NotImplementedError(f"The method for calculating the score of {task} is not Implemented.")
#
#     return score

@record
def main():
    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed + args.rank)
    # if args.rank==0:
    #     wandb.login(key="you_api_key")
    #     wandb.init(project="WebVideosVLN", name=args.output_dir.split('/')[-1])
    #     wandb.config.update(args)
    #
    #     flat_global_cfg = flatten_dict(global_cfg)
    #     wandb.config.update(flat_global_cfg)
    if args.enable_wandb and args.rank == 0:
        # 添加mode参数设置dryrun模式
        wandb.login(key="449b920ea50eef9966bf3bd5108550ddc3cab5c4")
        run = wandb.init(
            project=args.wandb_project,
            name=args.output_dir.split('/')[-1],
            mode="dryrun"  # offline mode
        )
        wandb.config.update(args)

        flat_global_cfg = flatten_dict(global_cfg)
        wandb.config.update(flat_global_cfg)


    ##################### DATASET #####################
    feat_db = create_feature_db(global_cfg.Feature.feature_database, global_cfg.Feature.image_feat_size, args)
    obj_feat_db = create_object_feature_db(global_cfg.Feature.object_database, global_cfg.Feature.obj_feat_size, args)
    # Initialize train dataloader
    if args.mode == "train":
        train_dataloaders, train_agents = create_dataloaders(
            args, global_cfg, logger,
            training=True, device=device_id, feat_db=feat_db, obj_feat_db=obj_feat_db, stage=args.stage
        )
    # Initialize val dataloader
    val_dataloaders, val_agents = create_dataloaders(
        args, global_cfg, logger,
        training=False, device=device_id, feat_db=feat_db, obj_feat_db=obj_feat_db, stage="multi"
    )

    # Model
    model = NavModel(args, logger, global_cfg.Model)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')

    model, optimizer, resume_from_epoch, lr_scheduler = dist_models(args, model, logger)

    # if args.enable_RL_A2C:
    #     from models.nav_model import Critic
    #     from tools.optims import dist_model_for_critic
    #     critic = Critic(args)
    #     from torch.nn.parallel import DistributedDataParallel as DDP
    #     if isinstance(model, DDP):
    #         critic = critic.to(model.module.model_type)
    #     else:
    #         critic = critic.to(model.model_type)
    #     critic, critic_optimizer, critic_resume_from_epoch, critic_lr_scheduler = dist_model_for_critic(args, critic, logger)
    #     assert resume_from_epoch == critic_resume_from_epoch, "critic and model should start from the same epoch"

    if args.mode=="test":
        logger.info("**************************** Test ****************************")
        results = val_one_epoch(
            args, global_cfg, model, optimizer, criterion, val_dataloaders, val_agents, resume_from_epoch, logger
        )
    elif args.mode == "train":
        logger.info("**************************** Train ****************************")

        best_results, best_score = None, None
        history_scores = []
        for epoch in range(resume_from_epoch, args.num_epochs):
            # # training
            # if args.enable_RL_A2C:
            #     train_one_epoch_RL(
            #         args, global_cfg, model, optimizer, lr_scheduler, criterion, train_dataloaders, train_agents, epoch,
            #         logger, stage=args.stage, critic=critic, critic_optimizer=critic_optimizer, critic_lr_scheduler=critic_lr_scheduler
            #     )
            # else:

            if args.alternate_post_training:
                if epoch % 2 == 0:
                    args.cot_output_as_supervision = True
                    args.replace_cot_gt_with_prob = True
                    args.replace_cot_gt_prob = 0.5
                    args.self_improving_cot = True
                    args.enable_self_select = True
                    args.self_select_loss_weight = 0.2
                else:
                    args.cot_output_as_supervision = False
                    args.replace_cot_gt_with_prob = False
                    args.self_improving_cot = False
                    args.enable_self_select = False

            train_one_epoch(
                args, global_cfg, model, optimizer, lr_scheduler, criterion, train_dataloaders, train_agents, epoch, logger, stage=args.stage
            )

            # evaluation
            results = val_one_epoch(
                args, global_cfg, model, optimizer, criterion, val_dataloaders, val_agents, epoch, logger
            )

            if args.rank==0:
                score = calc_overall_score(results, global_cfg)
                history_scores.append(score)
                should_save_checkpoint = False
                if args.enable_wandb:
                    wandb.log({"epoch": epoch, f"overall_score": score}, step=(epoch + 1) * args.num_steps_per_epoch - 1)

                if best_results is None or score > best_score:
                    best_results = results
                    best_score = score
                    should_save_checkpoint = args.max_saved_checkpoints > 0

                logger.info(f"Current Score: {score}")
                logger.info(f"Best Score: {best_score}")

                if args.stage=='multi':
                    # Save the best
                    if should_save_checkpoint:
                        if len(history_scores) > args.max_saved_checkpoints:
                            sorted_scores = sorted(enumerate(history_scores), key=lambda x: x[1], reverse=True)

                            remove_epoch = sorted_scores[args.max_saved_checkpoints][0]
                            remove_model_path = Path(args.output_dir) / f"epoch_{remove_epoch}.pt"
                            if os.path.exists(remove_model_path):
                                os.remove(remove_model_path)
                                logger.info(f"Remove Checkpoint at Epoch {remove_epoch}...")
                            if args.enable_lora:
                                remove_lora_dir = Path(args.output_dir) / f"lora-adapters_{remove_epoch}"
                                if args.enable_lora:
                                    if os.path.exists(remove_lora_dir):
                                        shutil.rmtree(remove_lora_dir)
                                        logger.info(f"Remove Lora at Epoch {remove_epoch}...")

                            # if args.enable_RL_A2C:
                            #     remove_critic_path = Path(args.output_dir) / f"critic_epoch_{remove_epoch}.pt"
                            #     if os.path.exists(remove_critic_path):
                            #         os.remove(remove_critic_path)
                            #         logger.info(f"Remove Critic Checkpoint at Epoch {remove_epoch}...")

                            model_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
                            save_checkpoint(model, model_path)

                            # if args.enable_RL_A2C:
                            #     critic_path = Path(args.output_dir) / f"critic_epoch_{epoch}.pt"
                            #     save_checkpoint(critic, critic_path)

                elif args.stage=='pretrain' and (epoch+1)%args.save_ckpt_per_epochs==0:
                    model_path = Path(args.output_dir) / f"pretrain_{epoch}.pt"
                    save_checkpoint(model, model_path)

              
            if args.save_latest_states:
                # if args.lora_finetune:
                #     state_dict = get_peft_state_maybe_zero_3(
                #         model.named_parameters(), args.lora_bias
                #     )
                #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                #         model.named_parameters()
                #     )
                #     if args.rank == 0 or args.rank == -1:
                #         # model.config.save_pretrained(args.output_dir)
                #         model.save_pretrained(args.output_dir, state_dict=state_dict)
                #         torch.save(non_lora_state_dict, os.path.join(args.output_dir, 'non_lora_trainables.bin'))
                # else:
                # Save the latest if args.save_latest_states is True
                if args.enable_lora:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        model.module.lang_model.save_pretrained(Path(args.output_dir) / "lora-adapters_last")
                    else:
                        model.lang_model.save_pretrained(Path(args.output_dir) / "lora-adapters_last")
                    # model.module.save_pretrained(Path(args.output_dir) / "lora-adapters_last", save_adapter=True, save_config=True)
                model_path = Path(args.output_dir) / f"latest.pt"
                save_checkpoint(model, model_path, optimizer, epoch, save_states=True)

                # if args.enable_RL_A2C:
                #     critic_path = Path(args.output_dir) / f"critic_latest.pt"
                #     save_checkpoint(critic, critic_path, critic_optimizer, epoch, save_states=True)

        # print best results
        if args.rank == 0:
            logger.info(f"Best Results:")
            logger.info(best_results)

if __name__ == '__main__':
    main()
