#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import os

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from split import split_video
from yolo_det.detect import start_det


def main():
    video_path = '50.mp4'
    temp_video_path = start_det(video_path, 50, 50)  # 和其它地方用到的start_det函数略有不同
    print("temp_video_path", temp_video_path)

    args = parse_args()
    args.cfg_file = "configs/SLOWFAST_4x16_R50.yaml"  # 配置文件路径
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    cfg.DEMO.INPUT_VIDEO = temp_video_path  # 覆盖来自配置文件的值

    csv_path = demo(cfg)
    print("csv_path:", csv_path)
    out_dir = split_video(cfg, csv_path, video_path)
    os.remove(temp_video_path)  # 删掉剪切的视频

    print("out_dir", out_dir)


if __name__ == "__main__":
    main()
