import itertools
import sys

import numpy as np
import cv2
from pydantic.experimental.pipeline import transform
from torch.utils.data import DataLoader

from utils.preprocessing import read_timestamps_from_file, process_events_hdr
from utils.vision_quality_compare import calculate_psnr, calculate_mse, calculate_ssim, calculate_lpips, test_dataset
from utils.load_hdf import get_dataset_shape
from PIL import Image
import os
import torch
import torchvision.models as models
from model.network import EHDR_network
from PIL import Image
import imageio as iio
import utils.HDR as hdr
from utils.vid2e import generate_timestamps, generate_events_loop, print_events
from train import SequenceDataset, CombinedLoss, RandomTransformNew
import torch.nn as nn


def check_npy(data_path, data_type: str = 'npy'):
    data = []
    if data_type == 'npy':
        data = np.load(data_path)
    print(data.shape)
    with np.load(data_path) as data:
        # print(data.shape)
        for key in data:
            # 获取数组
            array = data[key]
            # nonzero = np.count_nonzero(array[1][2])
            print(f"Array under key '{key}':\n{array.shape}\n")

    timestamps_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Exposure Left.txt'
    timestamps_pair = read_timestamps_from_file(timestamps_path)
    print(f'the average exposure time is: {np.mean([int(pair[1]) - int(pair[0]) for pair in timestamps_pair])}')


def npy_to_image(data_path, out_name, top, bottom, left, right, type: str = 'npy'):
    if type == 'npy':
        img = np.load(data_path)
    else:
        img = np.load(data_path)['data']
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    print(img[:, :, 0:3])
    # cv2.imshow('Processed Image', cv2.cvtColor(img[:, :, 0:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join('temp', out_name),
                cv2.cvtColor((img[top:bottom, left:right, 0:3] * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
    # print('press \'q\' or \'Esc\' to quit')
    # k = cv2.waitKey(0)
    # if k == 27 or k == ord('q'):  # 按下 ESC(27) 或 'q' 退出
    #     cv2.destroyAllWindows()


def crop_last_row(image_path, output_path):
    # 打开图片
    with Image.open(image_path) as img:
        # 获取图片尺寸
        width, height = img.size
        # 设置裁剪区域，裁剪掉最后一行像素
        cropped_image = img.crop((0, 0, width, height - 1))
        # 保存裁剪后的图片到新路径
        cropped_image.save(output_path)


def process_images_crop(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            original_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path,
                                         f"{os.path.splitext(filename)[0]}_cropped{os.path.splitext(filename)[1]}")
            crop_last_row(original_file_path, new_file_path)
            print(f"Processed {filename}, saved as {new_file_path.split('/')[-1]}")


def get_model_param_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")


def image_quality(img_path, correct_img_path):
    print(f'PSNR: {calculate_psnr(img_path, correct_img_path, False)}')
    print(f'MSE: {calculate_mse(img_path, correct_img_path, False)}')
    print(f'SSIM: {calculate_ssim(img_path, correct_img_path, False)}')
    print(f'LPIPS: {calculate_lpips(img_path, correct_img_path, False)}')


def normalize_to_8_bit(img):
    return (img * 255).astype('uint8')


def normalize_to_16_bit(img):
    return (img * 65535).astype('uint16')


def test_hdr():
    image_path = '/Volumes/CenJim/train data/dataset/HDM_HDR/smith_welding/smith_welding_249519.tif'
    img = iio.v3.imread(image_path)
    img = hdr.normalize_hdr(img, 16)
    img = hdr.pq_2_linear(img)
    img = hdr.rec2020_2_sRGB(img)
    exposure_time = hdr.histogram_based_exposure(img, target_percentile=99, target_value=0.9, gamma=2.2, tol=0.01,
                                                 max_iter=100)
    img = hdr.change_exposure(img, 1)
    img = hdr.apply_gamma(img, exposure_time, 2.2)
    img = normalize_to_8_bit(img)  # 转换为0-255的整数范围
    image = Image.fromarray(img)
    image.save('temp/output_image_sRGB.png')
    print(exposure_time)
    print(img.shape)


def crop_tiff(data_path, top, bottom, left, right):
    image = iio.v3.imread(data_path)
    image = image[10:-10, 10:-10]
    image = image[top:bottom, left:right]
    iio.v3.imwrite(os.path.join('temp', 'ground_truth_cropped.tif'), image)


def load_model_with_corrected_keys(model, state_dict_path):
    # 加载 state_dict
    state_dict = torch.load(state_dict_path)

    # 获取模型的现有 state_dict 的键
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())

    # 处理缺少的键
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys

    # 如果存在缺少或多余的键
    if missing_keys or unexpected_keys:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # 尝试修正 keys
        corrected_state_dict = {}
        for key in state_dict_keys:
            new_key = key.replace("module.", "")  # 去掉前缀 "module."
            if new_key in model_keys:
                corrected_state_dict[new_key] = state_dict[key]
            else:
                corrected_state_dict[key] = state_dict[key]

        # 更新 state_dict
        state_dict = corrected_state_dict

    # 加载修正后的 state_dict 到模型
    model.load_state_dict(state_dict, strict=True)


def loss_test(model_root_dir, test_data_dir):
    crop_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    trained_models = [os.path.join(model_root_dir, d) for d in sorted(os.listdir(model_root_dir)) if d.split('.')[-1] == 'pth']
    dataset = SequenceDataset(test_data_dir, transform=RandomTransformNew(256), hdr=True)  # Placeholder for your dataset class
    print(f'length of dataset: {len(dataset)}')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss_fun = CombinedLoss()
    model = EHDR_network(event_shape=(crop_size, crop_size), num_feat=64, num_frame=3)
    model = nn.DataParallel(model)
    model.to(device)
    for i, trained_model in enumerate(trained_models):
        load_model_with_corrected_keys(model, trained_model)
        # model.load_state_dict(torch.load(trained_model))
        model.eval()
        loss_avg = 0
        n = 0
        for j, (ldr_1, ldr_2, ldr_3, events_1, events_2, hdr) in enumerate(itertools.islice(dataloader, 0, None, 3)):
        # for j in range(0, len(dataloader), 3):
        # for j, (ldr_1, ldr_2, ldr_3, events_1, events_2, hdr) in enumerate(dataloader):
            # if j % 3 != 0:
            #     continue

            # print(f'num of iteration: {i}')
            n += 1
            ldr_1, ldr_2, ldr_3, events_1, events_2 = ldr_1.to(device), ldr_2.to(device), ldr_3.to(
                device), events_1.to(device), events_2.to(device)
            output = model(ldr_2, ldr_1, ldr_3, events_1, events_2).cpu().detach()

            # print(f'output: {output}')
            # print(f'hdr: {hdr}')
            loss = loss_fun(output, hdr)
            loss_avg += (loss - loss_avg) / n
            if j % 10 == 0:
                print(f'Model {i}, iteration {j} loss: {loss}')
                sys.stdout.flush()
            # model.reset_states()  # Reset states if your model has this functionality

        print(f"Model {i}, Loss: {loss_avg}")
        sys.stdout.flush()


if __name__ == '__main__':
    # img_dir = '/localdisk/home/s2491540/HDM_HDR/train/Carousel_Fireworks_02'
    # timestamp_dir = '/localdisk/home/s2491540/HDM_HDR/train/Carousel_Fireworks_02_timestamps.txt'
    #
    # generate_timestamps(25, 0, img_dir, timestamp_dir)

    # print_events('/home/s2491540/dataset/HDM_HDR/train/events_data_all.npz')

    data_path = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_02/sequence/ldr_images/poker_fullshot_046308_4.npz'
    out_name = 'pfmid.tif'
    # npy_to_image(data_path, out_name, 230, 830, 650, 1250, 'npz')
    npy_to_image(data_path, out_name, 0, 1060, 0, 1900, 'npz')
    # # ground_truth_path = '/home/s2491540/dataset/HDM_HDR/train/showgirl_02_301966.tif'
    # # crop_tiff(ground_truth_path, 230, 830, 650, 1250)

    # events = np.load('/home/s2491540/dataset/HDM_HDR/sequences/showgirl_01/events/000000_1.npz')
    # for key in events:
    #     event = events[key]
    # print(0)

    # model = EHDR_network((1060, 1900))
    # get_model_param_num(model)

    # process hdr events and save to a path
    # event_file = '/localdisk/home/s2491540/HDM_HDR/train/Carousel_Fireworks_02_events.npz'
    # output_folder = '/home/s2491540/dataset/HDM_HDR/sequences_not_for_train/Carousel_Fireworks_02/events'
    # image_timestamps_path = '/localdisk/home/s2491540/HDM_HDR/train/Carousel_Fireworks_02_timestamps.txt'
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # process_events_hdr(event_file, output_folder, image_timestamps_path, 1900, 1060, 5, 5, device, True, 'npz')

    model_dir = '/home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/pretrained_models/2.1-trained_on_8_sequences'
    data_dir = '/localdisk/home/s2491540/HDM_HDR/sequences'
    loss_test(model_dir, data_dir)

