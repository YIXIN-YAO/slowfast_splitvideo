#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os.path

import numpy as np
import time
import torch
import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


def mkdir_safe(path):
    """
    Creates a directory if it does not exist.
    :param path: path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def run_demo(cfg, frame_provider, model):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    # logger.info("Run demo with config:")
    # logger.info(cfg)
    # common_classes = (
    #     cfg.DEMO.COMMON_CLASS_NAMES
    #     if len(cfg.DEMO.LABEL_FILE_PATH) != 0
    #     else None
    # )

    # video_vis = VideoVisualizer(
    #     num_classes=cfg.MODEL.NUM_CLASSES,
    #     class_names_path=cfg.DEMO.LABEL_FILE_PATH,
    #     top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
    #     thres=cfg.DEMO.COMMON_CLASS_THRES,
    #     lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
    #     common_class_names=common_classes,
    #     colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
    #     mode=cfg.DEMO.VIS_MODE,
    # )
    #
    # async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    # if cfg.NUM_GPUS <= 1:
    #     model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    # else:
    #     model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        start = time.time()
        if cfg.DEMO.THREAD_ENABLE:
            frame_provider = ThreadVideoManager(cfg)
        else:
            frame_provider = VideoManager(cfg)  # 用的这一个
        model = get_model(cfg)
        content_list = []
        for task in tqdm.tqdm(run_demo(cfg, frame_provider, model), dynamic_ncols=True):
            frame_provider.display(task)
            # print("task:", task.action_preds)
            content_list.append([int(task.id), int(torch.argmax(task.action_preds[0])),
                                round(torch.max(task.action_preds[0]).item(), 2)])  # 8 6 0.48

        video_name = cfg.DEMO.INPUT_VIDEO
        _path = os.path.basename(video_name).split('.')[0] + '.csv'

        res_path = os.path.join(cfg.DEMO.CSV_DIR_PATH, _path)
        mkdir_safe(cfg.DEMO.CSV_DIR_PATH)
        write_result(content_list, res_path)

        frame_provider.join()
        frame_provider.clean()
        logger.info("Finish demo in: {}".format(time.time() - start))

        return res_path


def get_model(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    # logger.info("Run demo with config:")
    # logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    return model


def write_result(content_list, res_path):
    """
    写csv
    :param content_list: 内容
    :param res_path: 路径
    :return:
    """
    import csv

    with open(res_path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f, delimiter=' ')

        for i in content_list:
            csv_writer.writerow([i[0], i[1], i[2]])



