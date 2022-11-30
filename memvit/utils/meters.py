#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import os
import pickle
from collections import defaultdict, deque

import memvit.datasets.ava_helper as ava_helper
import memvit.utils.distributed as du
import memvit.utils.logging as logging
import memvit.utils.metrics as metrics
import memvit.utils.misc as misc
import numpy as np
import torch
from fvcore.common.timer import Timer
from iopath.common.file_io import g_pathmgr
from memvit.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)
from sklearn.metrics import average_precision_score

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE)
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(cfg, mode == "train")
        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True, data_dir=None):
        """
        Calculate and log the final AVA metrics.
        """
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats, self.output_dir)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            logging.log_json_stats(stats, self.output_dir)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
        frame_level=False,
        annotations=None,
        save_kinetics_preds=False,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.cfg = cfg
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        self.save_kinetics_preds = save_kinetics_preds

        self.num_cls = num_cls
        if not isinstance(num_cls, (tuple, list)):
            num_cls = [num_cls]

        # Initialize tensors.
        self.video_preds = [torch.zeros((num_videos, nc)) for nc in num_cls]
        if multi_label:
            for preds in self.video_preds:
                preds -= 1e10

        self.video_labels = [
            (
                torch.zeros((num_videos, nc))
                if multi_label
                else torch.zeros((num_videos)).long()
            )
            for nc in num_cls
        ]

        self.all_preds = [[] for _ in range(num_videos)]
        self.frame_level = frame_level

        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        for preds in self.video_preds:
            preds.zero_()
            if self.multi_label:
                preds -= 1e10

        for labels in self.video_labels:
            labels.zero_()

        self.all_preds = [[] for _ in range(len(self.all_preds))]

    def update_stats(self, preds, labels, clip_ids, ann_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for i, (
            one_preds,
            one_labels,
            one_video_preds,
            one_video_labels,
        ) in enumerate(zip(preds, labels, self.video_preds, self.video_labels)):
            for ind in range(one_preds.shape[0]):
                vid_id = int(clip_ids[ind]) // self.num_clips
                if one_video_labels[vid_id].sum() > 0:
                    assert torch.equal(
                        one_video_labels[vid_id].type(torch.FloatTensor),
                        one_labels[ind].type(torch.FloatTensor),
                    )
                one_video_labels[vid_id] = one_labels[ind]

                if self.save_kinetics_preds:
                    self.all_preds[vid_id].append(one_preds[ind].detach().cpu())

                if self.ensemble_method == "sum":
                    one_video_preds[vid_id] += one_preds[ind]
                elif self.ensemble_method == "max":
                    one_video_preds[vid_id] = torch.max(
                        one_video_preds[vid_id], one_preds[ind]
                    )
                else:
                    raise NotImplementedError(
                        "Ensemble Method {} is not supported".format(
                            self.ensemble_method
                        )
                    )
                if i == 0:
                    self.clip_count[vid_id] += 1
                    if ann_ids:
                        if self.ann_ids[vid_id] is not None:
                            assert self.ann_ids[vid_id] == ann_ids[ind]
                        self.ann_ids[vid_id] = ann_ids[ind]

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5), data_dir=None):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            for i, (video_preds, video_labels) in enumerate(
                zip(self.video_preds, self.video_labels)
            ):
                num_topks_correct = metrics.topks_correct(video_preds, video_labels, ks)
                topks = [(x / video_preds.size(0)) * 100.0 for x in num_topks_correct]
                assert len({len(ks), len(topks)}) == 1
                for k, topk in zip(ks, topks):
                    self.stats["task{}_top{}_acc".format(i, k)] = "{:.{prec}f}".format(
                        topk, prec=2
                    )
        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None

        num_tasks = max(len(cfg.MODEL.NUM_CLASSES_LIST), 1)

        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = [ScalarMeter(cfg.LOG_PERIOD) for _ in range(num_tasks)]
        self.mb_top5_err = [ScalarMeter(cfg.LOG_PERIOD) for _ in range(num_tasks)]
        # Number of misclassified examples.
        self.num_top1_mis = [0] * num_tasks
        self.num_top5_mis = [0] * num_tasks
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        for err in self.mb_top1_err:
            err.reset()
        for err in self.mb_top5_err:
            err.reset()
        self.num_top1_mis = [0] * len(self.num_top1_mis)
        self.num_top5_mis = [0] * len(self.num_top5_mis)
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr

        if isinstance(mb_size, list):
            assert len(set(mb_size)) == 1
            mb_size = mb_size[0]

        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            for i, (err1, err5, mb1, mb5) in enumerate(
                zip(top1_err, top5_err, self.mb_top1_err, self.mb_top5_err)
            ):
                mb1.add_value(err1)
                mb5.add_value(err5)
                # Aggregate stats
                self.num_top1_mis[i] += err1 * mb_size
                self.num_top5_mis[i] += err5 * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            for i, (mb1, mb5) in enumerate(zip(self.mb_top1_err, self.mb_top5_err)):
                stats[f"task{i}_top1_err"] = mb1.get_win_median()
                stats[f"task{i}_top5_err"] = mb5.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            for i in range(len(self.num_top1_mis)):
                stats[f"task{i}_top1_err"] = self.num_top1_mis[i] / self.num_samples
                stats[f"task{i}_top5_err"] = self.num_top5_mis[i] / self.num_samples
            stats["loss"] = self.loss_total / self.num_samples
        logging.log_json_stats(stats, self.output_dir)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()

        num_tasks = max(len(cfg.MODEL.NUM_CLASSES_LIST), 1)

        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = [ScalarMeter(cfg.LOG_PERIOD) for _ in range(num_tasks)]
        self.mb_top5_err = [ScalarMeter(cfg.LOG_PERIOD) for _ in range(num_tasks)]
        # Min errors (over the full val set).
        self.min_top1_err = [100.0] * num_tasks
        self.min_top5_err = [100.0] * num_tasks
        # Number of misclassified examples.
        self.num_top1_mis = [0] * num_tasks
        self.num_top5_mis = [0] * num_tasks
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        for err in self.mb_top1_err:
            err.reset()
        for err in self.mb_top5_err:
            err.reset()
        self.num_top1_mis = [0] * len(self.num_top1_mis)
        self.num_top5_mis = [0] * len(self.num_top5_mis)
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """

        if isinstance(mb_size, list):
            assert len(set(mb_size)) == 1
            mb_size = mb_size[0]

        for i, (err1, err5, mb1, mb5) in enumerate(
            zip(top1_err, top5_err, self.mb_top1_err, self.mb_top5_err)
        ):
            mb1.add_value(err1)
            mb5.add_value(err5)
            self.num_top1_mis[i] += err1 * mb_size
            self.num_top5_mis[i] += err5 * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            for i, (mb1, mb5) in enumerate(zip(self.mb_top1_err, self.mb_top5_err)):
                stats[f"task{i}_top1_err"] = mb1.get_win_median()
                stats[f"task{i}_top5_err"] = mb5.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            for i in range(len(self.num_top1_mis)):
                top1_err = self.num_top1_mis[i] / self.num_samples
                top5_err = self.num_top5_mis[i] / self.num_samples
                self.min_top1_err[i] = min(self.min_top1_err[i], top1_err)
                self.min_top5_err[i] = min(self.min_top5_err[i], top5_err)

                stats[f"task{i}_top1_err"] = top1_err
                stats[f"task{i}_top5_err"] = top5_err
                stats[f"task{i}_min_top1_err"] = self.min_top1_err[i]
                stats[f"task{i}_min_top5_err"] = self.min_top5_err[i]

        logging.log_json_stats(stats, self.output_dir)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
