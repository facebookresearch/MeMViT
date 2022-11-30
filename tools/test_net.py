#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import itertools
import os
import pickle
import random
from logging import raiseExceptions

import memvit.utils.checkpoint as cu
import memvit.utils.distributed as du
import memvit.utils.logging as logging
import memvit.utils.misc as misc
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from memvit.datasets import loader
from memvit.models import build_model
from memvit.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            if isinstance(labels, (list,)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda()
            else:
                labels = labels.cuda()

            video_idx = video_idx.cuda()

            if cfg.DETECTION.ENABLE:
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["video_name"], meta["boxes"])

            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            metadata = metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs, meta["video_name"])

            ann_ids = None
            if cfg.MVIT.FRAME_LEVEL:
                video_id = meta["video_id"]
                start_frame = meta["start_frame"]
                stop_frame = meta["stop_frame"]

            if len(cfg.MODEL.NUM_CLASSES_LIST) > 1:
                if cfg.MEMVIT.ENABLE:
                    assert len(labels) == len(cfg.MODEL.NUM_CLASSES_LIST)
                else:
                    assert len(labels.shape) == 2
                    labels = [labels[:, i].contiguous() for i in range(labels.shape[1])]
            else:
                preds = [preds]
                labels = [labels]

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)
                labels = du.all_gather(labels)
                [video_idx] = du.all_gather([video_idx])

                if cfg.MVIT.FRAME_LEVEL:
                    video_id = list(
                        itertools.chain.from_iterable(du.all_gather_unaligned(video_id))
                    )
                    start_frame = torch.cat(du.all_gather_unaligned(start_frame), dim=0)
                    stop_frame = torch.cat(du.all_gather_unaligned(stop_frame), dim=0)

            if cfg.NUM_GPUS:
                preds = [pred.cpu() for pred in preds]
                labels = [label.cpu() for label in labels]
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            if cfg.MVIT.FRAME_LEVEL:
                test_meter.update_frame_level_stats(
                    [pred.detach() for pred in preds],
                    video_id,
                    start_frame,
                    stop_frame,
                )
            else:
                test_meter.update_stats(
                    [pred.detach() for pred in preds],
                    [label.detach() for label in labels],
                    video_idx.detach(),
                    ann_ids=ann_ids,
                )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    test_meter.finalize_metrics(data_dir=cfg.DATA.PATH_TO_DATA_DIR)

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = [p.clone().detach() for p in test_meter.video_preds]
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

        if cfg.TEST.SAVE_KINETICS_PREDS:
            save_path = os.path.join(cfg.OUTPUT_DIR, "kinetics_preds.pkl")

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([test_meter.all_preds, all_labels], f)
            logger.info("Successfully saved prediction results to {}".format(save_path))

    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        if cfg.MEMVIT.ENABLE:
            assert (
                test_loader.dataset.real_num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )  # before padding.
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            cfg,
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES_LIST
            if cfg.MODEL.NUM_CLASSES_LIST
            else cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
            frame_level=cfg.MVIT.FRAME_LEVEL,
            annotations=(
                test_loader.dataset.load_annotations(test_loader.dataset.path_to_file)
                if cfg.MVIT.FRAME_LEVEL
                else None
            ),
            save_kinetics_preds=cfg.TEST.SAVE_KINETICS_PREDS,
        )
    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg)
