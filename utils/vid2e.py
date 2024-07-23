import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
import gc

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
    print("Loading images")
    images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE)[10:-10, 10:-10] for f in images])

    log_images = np.log(images.astype("float32") / 255 + 1e-4)

    # generate torch tensors
    device = "cuda:0"
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
    image_pattern = os.path.join(image_dir, '*.tif')
    image_files = sorted(glob.glob(image_pattern))
    timestamps_s = np.genfromtxt(timestamps_file)
    timestamps_ns = (timestamps_s * 1e9).astype("int64")
    for i in range(0, len(image_files), step):
        image_temp = image_files[i:i + step]
        timestamp_temp = timestamps_ns[i:i + step]
        events_dict = generate_events(esim, image_temp, timestamp_temp)
        dicts.append(events_dict)

    for key in dicts[0]:
        merged_dict[key] = []
    for d in dicts:
        for key, value in d.items():
            # 如果merged_dict[key]为空，则直接赋值，否则在现有数组后追加
            if len(merged_dict[key]) == 0:
                merged_dict[key] = value
            else:
                merged_dict[key] = np.concatenate((merged_dict[key], value))

    save_file = os.path.join(save_dir, 'events_data_all.npz')
    np.savez_compressed(save_file, x=merged_dict['x'], y=merged_dict['y'], p=merged_dict['p'], t=merged_dict['t'])


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
    image_dir = '/home/s2491540/dataset/HDM_HDR/train/showgirl_01'
    timestamps_file = '/home/s2491540/dataset/HDM_HDR/train/showgirl_01_timestamps.txt'
    save_dir = '/home/s2491540/dataset/HDM_HDR/sequences/showgirl_01/events'
    generate_events(image_dir, timestamps_file, save_dir, 0.2, 0.2)
