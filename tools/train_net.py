#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import pprint
import random

import memvit.models.losses as losses
import memvit.models.optimizer as optim
import memvit.utils.checkpoint as cu
import memvit.utils.distributed as du
import memvit.utils.logging as logging
import memvit.utils.metrics as metrics
import memvit.utils.misc as misc
import numpy as np
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from memvit.datasets import loader
from memvit.datasets.mixup import MixUp
from memvit.models import build_model
from memvit.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from memvit.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()

    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    accumulated_losses = []
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            if isinstance(labels, (list, tuple)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda()
            else:
                labels = labels.cuda()

            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = (
                            val[i].cuda(non_blocking=True)
                            if isinstance(val[i], torch.Tensor)
                            else val[i]
                        )
                else:
                    meta[key] = (
                        val.cuda(non_blocking=True)
                        if isinstance(val, torch.Tensor)
                        else val
                    )

        if len(cfg.MODEL.NUM_CLASSES_LIST) > 1:
            if cfg.MEMVIT.ENABLE:
                assert len(labels) == len(cfg.MODEL.NUM_CLASSES_LIST)
            else:
                assert len(labels.shape) == 2
                labels = [labels[:, i] for i in range(labels.shape[1])]

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            no_grad = False

            if cfg.DETECTION.ENABLE:
                preds = forward(
                    model,
                    (inputs, meta["video_name"], meta["boxes"]),
                    no_grad=no_grad,
                )
            else:
                preds = forward(model, (inputs, meta["video_name"]), no_grad=no_grad)

            if no_grad:
                train_meter.iter_toc()
                train_meter.iter_tic()
                continue

            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            if len(cfg.MODEL.NUM_CLASSES_LIST) > 1:
                preds = [pred.reshape((-1, pred.shape[-1])) for pred in preds]
                labels = [label.reshape((-1,)) for label in labels]
                loss = sum(
                    loss_fun(preds[task_idx], labels[task_idx].clone())
                    for task_idx in range(len(cfg.MODEL.NUM_CLASSES_LIST))
                )
            else:
                if not cfg.MIXUP.ENABLE and not cfg.DETECTION.ENABLE:
                    preds = preds.reshape((-1, preds.shape[-1]))
                    labels = labels.reshape((-1,))

                loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
        else:
            top1_errs, top5_errs, err_counts = [], [], []
            if not cfg.DATA.MULTI_LABEL:
                if not len(cfg.MODEL.NUM_CLASSES_LIST) > 1:
                    preds = [preds]
                    labels = [labels]

                for one_preds, one_labels in zip(preds, labels):
                    # Compute the errors.
                    top1_err, top5_err, err_count = metrics.compute_err(
                        one_preds, one_labels
                    )
                    top1_err *= err_count
                    top5_err *= err_count

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err, err_count = du.all_reduce(
                            [top1_err, top5_err, err_count]
                        )
                    if err_count > 0:
                        top1_err /= err_count
                        top5_err /= err_count

                    # Copy the stats from GPU to CPU (sync point).
                    top1_errs.append(top1_err.item())
                    top5_errs.append(top5_err.item())
                    err_counts.append(err_count.item())

            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(
                top1_errs,
                top5_errs,
                loss,
                lr,
                [c * max(cfg.NUM_GPUS, 1) for c in err_counts]
                if err_counts
                else inputs[0].size(0) * max(cfg.NUM_GPUS, 1),
            )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    if cfg.MEMVIT.ENABLE:
        misc.clear_memory(model, cfg)

    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):

        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            if isinstance(labels, (list,)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda()
            else:
                labels = labels.cuda()

            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = (
                            val[i].cuda(non_blocking=True)
                            if isinstance(val[i], torch.Tensor)
                            else val[i]
                        )
                else:
                    meta[key] = val.cuda(non_blocking=True)

        if len(cfg.MODEL.NUM_CLASSES_LIST) > 1:
            if cfg.MEMVIT.ENABLE:
                assert len(labels) == len(cfg.MODEL.NUM_CLASSES_LIST)
            else:
                assert len(labels.shape) == 2
                labels = [labels[:, i] for i in range(labels.shape[1])]

        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["video_name"], meta["boxes"])

            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs, video_names=meta["video_name"])

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                if not len(cfg.MODEL.NUM_CLASSES_LIST) > 1:
                    preds = [preds]
                    labels = [labels]

                top1_errs, top5_errs, err_counts = [], [], []
                for one_preds, one_labels in zip(preds, labels):
                    one_preds = one_preds.reshape((-1, one_preds.shape[-1]))
                    one_labels = one_labels.reshape((-1,))

                    top1_err, top5_err, err_count = metrics.compute_err(
                        one_preds, one_labels
                    )
                    top1_err *= err_count
                    top5_err *= err_count

                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err, err_count = du.all_reduce(
                            [top1_err, top5_err, err_count]
                        )

                    if err_count > 0:
                        top1_err /= err_count
                        top5_err /= err_count

                    # Copy the errors from GPU to CPU (sync point).
                    top1_errs.append(top1_err.item())
                    top5_errs.append(top5_err.item())
                    err_counts.append(err_count.item())

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_errs,
                    top5_errs,
                    [c * max(cfg.NUM_GPUS, 1) for c in err_counts]
                    if err_counts
                    else inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        if cfg.MEMVIT.ENABLE:
            misc.clear_memory(model, cfg)
            random.seed(cur_epoch)
            if "ava" in cfg.TRAIN.DATASET:
                train_loader.dataset._load_data(cfg, cur_epoch=cur_epoch)

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg)
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)


def forward(model, inputs, no_grad):
    if no_grad:
        with torch.no_grad():
            return model(*inputs)
    else:
        return model(*inputs)


def mask_tail(preds, labels, tail_actions, beta):
    B = preds.shape[0]
    tail = torch.stack([tail_actions.to(preds.device)] * B)
    for i in range(B):
        tail[i, labels[i]] = 0
    tail = tail * torch.bernoulli(torch.full(tail.shape, beta, device=preds.device))
    tail = tail.bool()
    preds_out = preds.clone()
    preds_out[tail] = -1e10
    return preds_out
