import sys

import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import gc
import torch.nn as nn
from tqdm import tqdm
import esim_torch


def generate_timestamps(frame_rate, begin_timestamp, img_dir, timestamps_dir):
    # 指定图像文件目录
    search_pattern = os.path.join(img_dir, '*.tif')
    img = sorted(glob.glob(search_pattern))

    # 帧率
    time_interval = 1 / frame_rate

    # 打开一个文件用于写入时间戳
    with open(timestamps_dir, 'w') as file:
        # 对于每一个图像文件，计算对应的时间戳并写入文件
        for i in range(len(img)):
            timestamp = i * time_interval + begin_timestamp
            # 写入时间戳到文件，每个时间戳后面跟一个换行符
            file.write(f"{timestamp:.9f}\n")

    print(f"时间戳文件已生成，共 {len(img)} 个时间戳。")


def generate_events(esim, images, timestamps):
    # esim = esim_torch.ESIM(contrast_threshold_neg=threshold_n,
    #                        contrast_threshold_pos=threshold_p,
    #                        refractory_period_ns=0)
    device = "cuda"
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     esim = nn.DataParallel(esim)
    # esim.to(device)
    print("Loading images")
    images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE)[10:-10, 10:-10] for f in images])

    log_images = np.log(images.astype("float32") / 255 + 1e-4)

    # generate torch tensors
    log_images = torch.from_numpy(log_images).to(device)
    timestamps_ns = torch.from_numpy(timestamps).to(device)

    # generate events with GPU support
    print("Generating events")
    events = esim.forward(log_images, timestamps_ns)
    events_dict = {k: v.cpu().numpy() for k, v in events.items()}

    del log_images, timestamps_ns, events
    torch.cuda.empty_cache()  # 清空 PyTorch 的 GPU 缓存
    gc.collect()  # 显式调用垃圾回收器
    return events_dict


def generate_events_loop(image_dir, timestamps_file, save_dir, threshold_p=0.2, threshold_n=0.2, step=100):
    esim = esim_torch.ESIM(contrast_threshold_neg=threshold_n,
                           contrast_threshold_pos=threshold_p,
                           refractory_period_ns=0)
    merged_dict = {}
    dicts = []
    print("Loading images")
    sys.stdout.flush()
    image_pattern = os.path.join(image_dir, '*.tif')
    image_files = sorted(glob.glob(image_pattern))
    timestamps_s = np.genfromtxt(timestamps_file)
    timestamps_ns = (timestamps_s * 1e9).astype("int64")
    for i in tqdm(range(0, len(image_files), step), total=len(range(0, len(image_files), step)),
                  desc='events generation'):
        if i > 0:
            image_temp = image_files[i - 1:i + step]
            timestamp_temp = timestamps_ns[i - 1:i + step]
        else:
            image_temp = image_files[i:i + step]
            timestamp_temp = timestamps_ns[i:i + step]
        events_dict = generate_events(esim, image_temp, timestamp_temp)
        dicts.append(events_dict)
        sys.stdout.flush()

    print('saving the events process')
    sys.stdout.flush()
    # for key in dicts[0].keys():
    #     merged_dict[key] = []
    # for d in dicts:
    #     for key, value in d.items():
    #         # 如果merged_dict[key]为空，则直接赋值，否则在现有数组后追加
    #         if len(merged_dict[key]) == 0:
    #             merged_dict[key] = value
    #         else:
    #             merged_dict[key] = np.concatenate((merged_dict[key], value))
    final_size = sum(len(d['p']) for d in dicts)
    # 预先分配内存
    for key in dicts[0].keys():
        if key == 't':
            merged_dict[key] = np.empty(final_size, dtype=np.int64)  # 或者根据你的数据类型调整
        else:
            merged_dict[key] = np.empty(final_size, dtype=np.int16)  # 或者根据你的数据类型调整

    # 使用索引填充数据
    for key in merged_dict.keys():
        start = 0
        for d in dicts:
            size = len(d[key])
            merged_dict[key][start:start + size] = d[key]
            start += size

    # sort as t
    print('sorting events indices')
    sys.stdout.flush()
    t_indices = np.argsort(merged_dict['t'])  # 获取排序后的索引数组
    print('events sorted')
    print('get the sorted output')
    sys.stdout.flush()
    merged_dict['x'] = merged_dict['x'][t_indices]  # 根据索引排序 x
    merged_dict['y'] = merged_dict['y'][t_indices]  # 根据索引排序 y
    merged_dict['p'] = merged_dict['p'][t_indices]  # 根据索引排序 p
    merged_dict['t'] = merged_dict['t'][t_indices]  # t 本身也需要排序
    print('sorted output got')
    print('savez')
    sys.stdout.flush()
    # save_file = os.path.join(save_dir, 'events_data_all.npz')
    save_file = save_dir
    np.savez_compressed(save_file, x=merged_dict['x'], y=merged_dict['y'], p=merged_dict['p'], t=merged_dict['t'])
    print('compressed saved')
    print('saving events process done')
    sys.stdout.flush()


def print_events(events_file):
    events = np.load(events_file)
    x = events['x']
    y = events['y']
    p = events['p']
    t = events['t']
    print(t[-10:])
    print(x[-10:])
    print(y[-10:])
    print(p[-10:])


if __name__ == "__main__":
    # img_dir = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_pf2'
    # timestamp_dir = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/pf2_timestamps.txt'
    # generate_timestamps(24, 0, img_dir, timestamp_dir)


    image_dir = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_pf2'
    timestamps_file = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/pf2_timestamps.txt'
    save_dir = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/pf2_events.npz'
    generate_events_loop(image_dir, timestamps_file, save_dir, 0.02, 0.02, 15)
