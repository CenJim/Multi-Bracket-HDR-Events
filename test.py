import numpy as np
import cv2
from utils.preprocessing import read_timestamps_from_file
from utils.vision_quality_compare import calculate_psnr, calculate_mse, calculate_ssim, calculate_lpips
from utils.load_hdf import get_dataset_shape
from PIL import Image
import os
import torch
import torchvision.models as models
from model.network import EHDR_network


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


def npy_to_image(data_path):
    img = np.load(data_path)
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    print(img[:, :, 0:3])
    # cv2.imshow('Processed Image', cv2.cvtColor(img[:, :, 0:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite('temp/hdr.png', cv2.cvtColor((img[:, :, 0:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
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
            new_file_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_cropped{os.path.splitext(filename)[1]}")
            crop_last_row(original_file_path, new_file_path)
            print(f"Processed {filename}, saved as {new_file_path.split('/')[-1]}")


def get_model_param_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")


if __name__ == '__main__':
    # img_path = '../temp/000001_2_cropped.png'
    # correct_img_path = '../temp/hdr_cropped.png'
    # print(f'PSNR: {calculate_psnr(img_path, correct_img_path, False)}')
    # print(f'MSE: {calculate_mse(img_path, correct_img_path, False)}')
    # print(f'SSIM: {calculate_ssim(img_path, correct_img_path, False)}')
    # print(f'LPIPS: {calculate_lpips(img_path, correct_img_path, False)}')
    model = EHDR_network(event_shape=(256, 256), num_feat=64, num_frame=3)
    get_model_param_num(model)

