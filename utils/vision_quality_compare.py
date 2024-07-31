import math

import torch
from imageio.config.plugins import module_name
from mpl_toolkits.mplot3d.proj3d import transform
from numpy.core.numeric import False_
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os
import numpy as np
import lpips
from train import SequenceDataset
import torchvision.transforms as transforms
from evaluation import inference
from torch.utils.data import DataLoader
from tqdm import tqdm


def histogram_normalization(img_dir, gray_flag: bool = True):
    if gray_flag:
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(img_dir, cv2.IMREAD_COLOR)

    # int to float
    image_float = image.astype(np.float32)

    min_val = np.min(image_float)
    max_val = np.max(image_float)

    # normalize
    normalized_image = 255 * (image_float - min_val) / (max_val - min_val)

    # float to int
    normalized_image = normalized_image.astype(np.uint8)

    return normalized_image


def preprocess_lpips(img_dir):
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image * 2 - 1
    image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]  # transform the image from H*W*3 to 1*3*H*W
    return torch.from_numpy(image)


def preprocess_lpips_avg(img_dir):
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image * 2 - 1
    image = np.transpose(image, (2, 0, 1))  # transform the image from H*W*3 to 3*H*W
    return image


def calculate_psnr(img_1_dir, img_2_dir, gray_flag: bool = True):
    img_correct = histogram_normalization(img_1_dir, gray_flag)
    img_compared = histogram_normalization(img_2_dir, gray_flag)
    # if gray_flag:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_GRAYSCALE)
    # else:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_COLOR)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_COLOR)
    psnr = compare_psnr(img_correct, img_compared)
    return psnr


def calculate_ssim(img_1_dir, img_2_dir, gray_flag: bool = True):
    img_correct = histogram_normalization(img_1_dir, gray_flag)
    img_compared = histogram_normalization(img_2_dir, gray_flag)
    # if gray_flag:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_GRAYSCALE)
    # else:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_COLOR)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_COLOR)
    if gray_flag:
        ssim = compare_ssim(img_correct, img_compared)
    else:
        ssim = compare_ssim(img_correct, img_compared, channel_axis=-1)
    return ssim


def calculate_mse(img_1_dir, img_2_dir, gray_flag: bool = True, channel_axis=2):
    img_correct = histogram_normalization(img_1_dir, gray_flag)
    img_compared = histogram_normalization(img_2_dir, gray_flag)
    # if gray_flag:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_GRAYSCALE)
    # else:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_COLOR)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_COLOR)
    mse = compare_mse(img_correct, img_compared)
    return mse


def calculate_lpips(img_1_dir, img_2_dir, gray_flag: bool = True):
    img_correct = preprocess_lpips(img_1_dir)
    img_compared = preprocess_lpips(img_2_dir)
    loss_fn_alex = lpips.LPIPS(net='alex')
    d = loss_fn_alex(img_correct, img_compared)
    return d[0][0][0][0]


def test_dataset(dataset_dir, hdr: bool, model_name, pretrain_models_path, device):
    dataset = SequenceDataset(dataset_dir, transform=transforms.RandomCrop(600), hdr=hdr, u_law_compress=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    psnr_mean = 0.0
    ssim_mean = 0.0
    mse_mean = 0.0
    lpips_mean = 0.0
    psnr_S = 0.0
    ssim_S = 0.0
    mse_S = 0.0
    lpips_S = 0.0
    correct_image_list = []
    compare_image_list = []
    n = 0
    for i, (ldr_1, ldr_2, ldr_3, events_1, events_2, hdr_true) in tqdm(enumerate(data_loader), total=len(data_loader),
                                                                       desc='test dataset'):
        ldr_1, ldr_2, ldr_3, events_1, events_2 = ldr_1.to(device), ldr_2.to(device), ldr_3.to(
            device), events_1.to(device), events_2.to(device)
        out_img = inference(model_name, pretrain_models_path,
                            (ldr_1, ldr_2, ldr_3, events_1, events_2),
                            hdr=hdr, compress='PQ')
        n += 1
        hdr_true = hdr_true.numpy().astype(np.float64).transpose(1, 2, 0)

        psnr_temp = compare_psnr(hdr_true, out_img)
        old_psnr_mean = psnr_mean
        psnr_mean += (psnr_temp - psnr_mean) / n
        psnr_S += (psnr_temp - old_psnr_mean) * (psnr_temp - psnr_mean)

        ssim_temp = compare_ssim(out_img, hdr_true, channel_axis=-1)
        old_ssim_mean = ssim_mean
        ssim_mean += (ssim_temp - ssim_mean) / n
        ssim_S += (ssim_temp - old_ssim_mean) * (ssim_temp - ssim_mean)

        mse_temp = compare_mse(out_img, hdr_true)
        old_mse_mean = mse_mean
        mse_mean += (mse_temp - mse_mean) / n
        mse_S += (mse_temp - old_mse_mean) * (mse_temp - mse_mean)
        # lpips_all = lpips_all + calculate_lpips(os.path.join(correct_img_directory, filename),
        #                                         os.path.join(directory, compare_files[(index + 1) * step - 1]))
        correct_image_list.append((hdr_true * 2 - 1).transpose(2, 0, 1))
        compare_image_list.append((out_img * 2 - 1).transpose(2, 0, 1))
    loss_fn_alex = lpips.LPIPS(net='alex')
    lpips_list = loss_fn_alex(torch.from_numpy(np.stack(correct_image_list)),
                              torch.from_numpy(np.stack(compare_image_list)))
    return {'psnr_mean': psnr_mean, 'ssim_mean': ssim_mean, 'mse_mean': mse_mean, 'lpips': float(torch.mean(lpips_list).item()),
            'psnr_std': math.sqrt(psnr_S / (n - 1)), 'ssim_std': math.sqrt(ssim_S / (n - 1)),
            'mse_std': math.sqrt(mse_S / (n - 1)), 'lpips_std': float(torch.std(lpips_list, unbiased=True).item())}


def calculate_average_quality(img_directory_list: list, correct_img_directory: str, step: int = 1):
    correct_files = [f for f in os.listdir(correct_img_directory) if
                     os.path.isfile(os.path.join(correct_img_directory, f)) and (
                             os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.bmp' or
                             os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.tif')]
    correct_files.sort()
    psnr_mean = 0.0
    ssim_mean = 0.0
    mse_mean = 0.0
    lpips_mean = 0.0
    psnr_S = 0.0
    ssim_S = 0.0
    mse_S = 0.0
    lpips_S = 0.0
    correct_image_list = []
    compare_image_list = []
    n = 0
    for directory in img_directory_list:
        compare_files = [f for f in os.listdir(directory) if
                         os.path.isfile(os.path.join(directory, f)) and (
                                 os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.bmp' or
                                 os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.tif')]
        compare_files.sort()
        for index, filename in enumerate(correct_files):
            n += 1
            psnr_temp = calculate_psnr(os.path.join(correct_img_directory, filename),
                                       os.path.join(directory, compare_files[(index + 1) * step - 1]))
            old_psnr_mean = psnr_mean
            psnr_mean += (psnr_temp - psnr_mean) / n
            psnr_S += (psnr_temp - old_psnr_mean) * (psnr_temp - psnr_mean)

            ssim_temp = calculate_ssim(os.path.join(correct_img_directory, filename),
                                       os.path.join(directory, compare_files[(index + 1) * step - 1]))
            old_ssim_mean = ssim_mean
            ssim_mean += (ssim_temp - ssim_mean) / n
            ssim_S += (ssim_temp - old_ssim_mean) * (ssim_temp - ssim_mean)

            mse_temp = calculate_mse(os.path.join(correct_img_directory, filename),
                                     os.path.join(directory, compare_files[(index + 1) * step - 1]))
            old_mse_mean = mse_mean
            mse_mean += (mse_temp - mse_mean) / n
            mse_S += (mse_temp - old_mse_mean) * (mse_temp - mse_mean)
            # lpips_all = lpips_all + calculate_lpips(os.path.join(correct_img_directory, filename),
            #                                         os.path.join(directory, compare_files[(index + 1) * step - 1]))
            correct_image_list.append(preprocess_lpips_avg(os.path.join(correct_img_directory, filename)))
            compare_image_list.append(
                preprocess_lpips_avg(os.path.join(directory, compare_files[(index + 1) * step - 1])))
    loss_fn_alex = lpips.LPIPS(net='alex')
    lpips_list = loss_fn_alex(torch.from_numpy(np.stack(correct_image_list)),
                              torch.from_numpy(np.stack(compare_image_list)))

    return {'psnr_mean': psnr_mean, 'ssim_mean': ssim_mean, 'mse_mean': mse_mean, 'lpips': float(torch.mean(lpips_list).item()),
            'psnr_std': math.sqrt(psnr_S / (n - 1)), 'ssim_std': math.sqrt(ssim_S / (n - 1)),
            'mse_std': math.sqrt(mse_S / (n - 1)), 'lpips_std': float(torch.std(lpips_list, unbiased=True).item())}


def rename_files_in_directory(directory, prefix="file", add_old=False):
    """
        Change the filenames in the specified directory, generating filenames in Arabic numerical order.

        Parameters.
        directory: path of the directory where the filename should be changed (string).
        prefix: the prefix of the new filename (string), default is "file".
        add_old: whether add the old filename as the prefix ot not.
    """
    try:
        # obtain all the files under the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()

        # rename
        for index, filename in enumerate(files):
            # get the file extensions
            extension = os.path.splitext(filename)[1]
            if add_old:
                new_name = f"{filename.split('.')[0]}{prefix}{index}{extension}"
            else:
                new_name = f"{prefix}{index}{extension}"
            # generate full directory
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

        print("all files renamed")
    except Exception as e:
        print(f"error occur: {e}")


# 示例用法
if __name__ == "__main__":
    # directory_path = '/Users/macbookpro/python_proj/vision_quality_compare/data/IJRR/EV2ID_index_20.png'
    # cv2.imwrite('/Users/macbookpro/python_proj/vision_quality_compare/data/IJRR/EV2ID_index_20_normalized.png',
    #             histogram_normalization(directory_path))
    dataset_dir = '/home/s2491540/dataset/HDM_HDR/sequences_not_for_train'
    model_name = 'EHDR_network'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = ''
    test_dataset(dataset_dir, hdr=True, model_name=module_name, pretrain_models_path=pretrained_model, device = device)
