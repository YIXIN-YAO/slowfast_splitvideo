import os
import cv2

import numpy as np
import matplotlib.pyplot as plt


def mkdir_safe(path):
    """
    Creates a directory if it does not exist.
    :param path: path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_video_stream(video_path, action, start_point, length, save_dir):
    """
    从原视频里获取视频切片，从第start_point x 64帧到第（start_point+length）x64帧
    :param video_path: 原视频地址
    :param action: 动作类别
    :param start_point: 切分起始点
    :param length: 动作长度，默认每一个点是64帧
    :param save_dir: 目标文件夹
    :return:
    """
    start = start_point * 64  # 起始帧
    end = (start_point + length) * 64  # 结束帧

    video_read = cv2.VideoCapture(video_path)

    fps = int(video_read.get(cv2.CAP_PROP_FPS))  # 保存视频的帧率
    frame_sum = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    video_write = cv2.VideoWriter(os.path.join(save_dir, action + '.avi'),
                                  cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

    i = 0
    while i < frame_sum:
        success, frame = video_read.read()
        if success:
            i += 1
            if start <= i <= end:
                video_write.write(frame)
            elif i > end:
                break
        else:
            break
    video_read.release()


def find(label_list, target, max_stride):
    """
    从一个序列中找到最长的连续target序列
    :param label_list:
    :param target:
    :param max_stride: 允许的最大步长，就是可以跳过max_stride-1个异常帧
    :return: 返回最长序列的头节点和长度
    """

    def is_connected(pos):
        for step in range(1, max_stride + 1):
            if pos + step == len(label_list):
                break  # 防止越界
            if label_list[pos + step] == target:
                return step
        return 0

    max_length = 0
    start_pos = -1
    for index, i in enumerate(label_list):

        if i == target:
            if index > 0 and label_list[index - 1] == target:
                continue
            else:  # 进入else的都是头节点
                length = 1
                while index != len(label_list) - 1 and is_connected(index):
                    length += is_connected(index)
                    index += is_connected(index)
                if length > max_length:
                    max_length = length
                    start_pos = index - max_length + 1
    return start_pos, max_length


def vis(label_list):
    """
    可视化
    :param label_list: 从csv提取出的label的list
    :return:
    """
    category_names = []
    cls_dict = {
        0: "diandi",
        1: "down",
        2: "duizhi",
        3: "pose",
        4: "tadi",
        5: "up",
        6: "woquan",
        7: "xiangxia",
        8: "zhengfan"
    }
    results = {
        'res': []
    }
    num = 0
    for i in range(len(label_list)):
        num += 1
        if i is not len(label_list) - 1:
            if label_list[i] != label_list[i + 1]:
                results['res'].append(num)
                category_names.append(cls_dict[label_list[i]])
                num = 0
        else:
            results['res'].append(num)
            category_names.append(cls_dict[label_list[i]])
            num = 0
    survey(results, category_names)
    # print(results['res'])
    # print(category_names)


def survey(results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    fig, ax = plt.subplots(figsize=(20, 2))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    r = 1
    g = 1
    b = 1
    color_dict = {
        "diandi": [0.6, 0.3, b, 1],
        "down": [r, g, 0, 1],
        "duizhi": [r, 0, b, 1],
        "pose": [0.9, 0, 0.5, 1],
        "tadi": [0, g, b, 1],
        "up": [0, g, 0.5, 1],
        "woquan": [0, 0.5, b, 1],
        "xiangxia": [0.48858131, 0.7799308, 0.39792388, 1],
        "zhengfan": [0.84805844, 0.66720492, 0.3502499, 1]
    }

    for i, colname in enumerate(category_names):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        color = color_dict[colname]
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    plt.savefig('./csv_res/temp.jpg')
    plt.show()
    # return fig, ax


def split_video(cfg, csv_path, raw_input_video):
    out_dir = os.path.join(cfg.DEMO.OUTPUT_FILE_DIR, os.path.basename(cfg.DEMO.INPUT_VIDEO).split('.')[0])
    mkdir_safe(out_dir)
    cls_dict = {
        0: "diandi",
        1: "down",
        2: "duizhi",
        3: "pose",
        4: "tadi",
        5: "up",
        6: "woquan",
        7: "xiangxia",
        8: "zhengfan"
    }

    label_list = list()
    with open(csv_path, "r") as f:
        for idx, line in enumerate(f.read().splitlines()):
            assert (len(line.split(' ')) == 3)
            task_id, label, score = line.split(' ')
            label_list.append(int(label))

        vis(label_list)  # 可视化
        max_stride = cfg.DEMO.SPLIT_MAX_STRIDE  # 这个参数可以通过做可视化，去调节。
        for target in range(9):
            start_point, length = find(label_list, target, max_stride)
            get_video_stream(raw_input_video, cls_dict[target], start_point, length, out_dir)

    return out_dir


if __name__ == "__main__":
    pass
