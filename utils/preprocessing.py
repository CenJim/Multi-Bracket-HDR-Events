import cv2
import numpy as np
import os
import glob

from tqdm import tqdm

from utils.load_hdf import get_dataset, get_event_offset, chunk_2d_array, chunk_2d_array_fix_num
from utils.representations import VoxelGrid
import torch
from PIL import Image


def inverse_crf(Z, gamma=2.2):
    """ 应用逆CRF（伽马校正的逆函数） """
    return np.power(Z, gamma)


def apply_inverse_crf_and_normalize(img, exposure_time, gamma=2.2):
    # 应用逆CRF
    img_linear = inverse_crf(img, gamma)

    # 除以曝光时间
    img_corrected = img_linear / exposure_time

    # 将结果重新缩放到0-255并转换为整数
    # img_corrected = np.clip(img_corrected * 255.0, 0, 255).astype(np.uint8)

    return img_corrected


def transfer_hdr_to_ldr(hdr_image, exposure_time, gamma=2.2):
    """ 将HDR图像转换为LDR图像 """
    # hdr_image = hdr_image / 255
    ldr_image = np.clip(np.power((hdr_image * exposure_time), 1 / gamma), 0.2, 0.8)
    return ldr_image


def concatenate_to_six_channels(image, exposure_time):
    compensated_image = inverse_crf(image, gamma=2.2) / exposure_time
    return np.dstack((image, compensated_image))


def resize_and_crop(img, resize_width=715, resize_height=536, center_x=338, center_y=301, crop_width=640,
                    crop_height=469):
    # 将图像降采样到715x536
    img_resized = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

    # 计算裁剪框的左上角坐标
    # 中心点 (338, 301) - 裁剪尺寸的一半 (320, 234.5)
    # 因为坐标必须是整数，使用round进行四舍五入
    left = round(center_x - crop_width / 2)
    top = round(center_y - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    # 根据计算得到的坐标裁剪图像
    img_cropped = img_resized.crop((left, top, right, bottom))

    # 返回裁剪后的图像
    return img_cropped


def normalize_image(image_pil):
    # 从pillow格式转换图像, 并映射到0～1
    img = np.array(image_pil)
    return img.astype(np.float32) / 255.0


def scale_value(image, min_val=0.2, max_val=0.8):
    return (image - min_val) / (max_val - min_val)


def process_images(source_folder, target_folder, supervised_folder, exposure_time_path, save_format: str = 'npy'):
    """
    遍历指定文件夹下的所有图片，对每个图片进行处理，并保存为.npy文件到新的文件夹。

    :param source_folder: 包含原始图片的文件夹路径
    :param target_folder: 保存.npy文件的目标文件夹路径
    :param exposure_time: 标准的曝光时间
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    if not os.path.exists(supervised_folder):
        os.makedirs(supervised_folder)
    processed_images = {}
    supervised_images = {}
    # 构建图片文件的搜索模式
    search_pattern = os.path.join(source_folder, '*.png')  # 假设图片是PNG格式
    image_files = glob.glob(search_pattern)

    exposure_time_pairs = read_timestamps_from_file(exposure_time_path)

    # 遍历所有图片文件
    i = 0
    for index, image_file in tqdm(enumerate(sorted(image_files)), total=len(image_files), desc='Processing images'):
        if i >= 3:
            i = 1
        else:
            i += 1

        image_pil = Image.open(image_file)
        image_pil = resize_and_crop(image_pil, resize_width=715, resize_height=536, center_x=338, center_y=301,
                                    crop_width=640, crop_height=469)
        image = normalize_image(image_pil)
        exposure_time = (np.float64(exposure_time_pairs[index][1]) - np.float64(
            exposure_time_pairs[index][0])) / 1000000
        # 对图片进行特定操作
        if i == 1:
            hdr_image = apply_inverse_crf_and_normalize(image, exposure_time)
            ldr_images = transfer_hdr_to_ldr(hdr_image, exposure_time / 3)
            processed_image = concatenate_to_six_channels(ldr_images, exposure_time / 3)
        elif i == 3:
            hdr_image = apply_inverse_crf_and_normalize(image, exposure_time)
            ldr_images = transfer_hdr_to_ldr(hdr_image, exposure_time * 3)
            processed_image = concatenate_to_six_channels(ldr_images, exposure_time * 3)
        else:
            ldr_image = np.clip(image, 0.2, 0.8)
            # image = scale_value(image, 0.2, 0.8)
            processed_image = concatenate_to_six_channels(ldr_image, exposure_time)

        image = image.transpose((2, 0, 1))
        processed_image = processed_image.transpose((2, 0, 1))
        # 生成目标文件路径
        base_name = os.path.basename(image_file)
        if save_format == 'npy':
            target_file = os.path.join(target_folder, os.path.splitext(base_name)[0] + f'_{i}.npy')

            # 保存处理后的图片数据为.npy文件
            np.save(target_file, processed_image)
            # print(f"Processed image saved to {target_file}")
            if i == 2:
                supervised_file = os.path.join(supervised_folder, os.path.splitext(base_name)[0] + '.npy')
                np.save(supervised_file, image)
        elif save_format == 'npz':
            base_name = os.path.basename(image_file)
            key = os.path.splitext(base_name)[0] + f'_{i}'
            processed_images[key] = processed_image
            # print("Processed image: " + key)
            if i == 2:
                key = os.path.splitext(base_name)[0]
                supervised_images[key] = image
        else:
            target_file = os.path.join(target_folder, os.path.splitext(base_name)[0] + f'_{i}.pt')
            # 保存处理后的图片数据为.pt文件
            torch.save(torch.from_numpy(processed_image), target_file)
            # print(f"Processed image saved to {target_file}")
            if i == 2:
                supervised_file = os.path.join(supervised_folder, os.path.splitext(base_name)[0] + '.pt')
                torch.save(torch.from_numpy(image), supervised_file)

    if save_format == 'npz':
        npz_target_file = os.path.join(target_folder, 'all_processed_images.npz')
        print('Saving processed npz')
        np.savez_compressed(npz_target_file, **processed_images)
        print(f"All processed images saved to {npz_target_file}")
        npz_supervised_file = os.path.join(supervised_folder, 'all_supervised_image.npz')
        print('Saving supervised npz')
        np.savez_compressed(npz_supervised_file, **supervised_images)
        print(f"All supervised images saved to {npz_supervised_file}")


def read_timestamps_from_file(file_path):
    # 初始化一个空列表，用于存储转换后的时间戳对
    timestamp_pairs = []

    # 打开并读取文件
    with open(file_path, 'r') as file:
        # 使用next函数来跳过第一行
        next(file)

        # 遍历文件中的每一行
        for line in file:
            # 去除行末的换行符，并以逗号分隔字符串，转换为元组
            parts = line.strip().split(',')
            if len(parts) == 2:  # 确保每行确实有两个元素
                # 将字符串转换为整数，并添加到列表中
                timestamp_pairs.append((int(parts[0]), int(parts[1])))

    return timestamp_pairs


def get_exposure_time(timestamp_pairs):
    exposure_timestamps = []
    for pair in timestamp_pairs:
        exposure_timestamps.append((pair[0] + pair[1]) / 2)
    return exposure_timestamps


def events_to_voxel_grid(x, y, p, t, voxel_grid: VoxelGrid, device: str = 'cpu'):
    t = (t - t[0]).astype('float32')
    t = (t / t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    return voxel_grid.convert(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(pol),
        torch.from_numpy(t),
        device)


def get_voxel_grid(events, height, width, number_chunk, num_bins, device):
    # chunked_events = chunk_2d_array(events, int(width * height * num_events_per_pixel))
    chunked_events = chunk_2d_array_fix_num(events, number_chunk)
    voxel_grid = VoxelGrid(num_bins, height, width, True)
    event_tensors = []
    for chunked_event in chunked_events:
        event_tensor = events_to_voxel_grid(chunked_event[:, 1], chunked_event[:, 2], chunked_event[:, 3],
                                            chunked_event[:, 0], voxel_grid, device)
        event_tensor = event_tensor[:, :469, :].cpu()
        event_tensors.append(event_tensor)
    return event_tensors


def process_events(source_folder, target_folder, image_timestamps, width, height, num_chunks, num_bins,
                   device, save_flag: bool = False, save_format: str = 'npy'):
    print('Loading the dataset...')
    events_dataset = get_dataset(source_folder)
    print('Dataset Loaded!')
    events_offset = get_event_offset(source_folder)
    exposure_timestamps = get_exposure_time(read_timestamps_from_file(image_timestamps))
    processed_events = {}
    i = 0
    events_length = len(events_dataset)
    events_chunks = []
    j = 0
    for index, exposure_timestamp in tqdm(enumerate(exposure_timestamps[:-1]), total=len(exposure_timestamps) - 1,
                                          desc='Events processing'):
        if j >= 3:
            j = 1
        else:
            j += 1
        events_chunk = []
        while i < events_length:
            if exposure_timestamp <= events_dataset[i][0] + events_offset <= exposure_timestamps[index + 1]:
                events_chunk.append(events_dataset[i])
                i += 1
            elif events_dataset[i][0] + events_offset > exposure_timestamps[index + 1]:
                break
            else:
                i += 1
        if j == 3:
            continue
        with torch.no_grad():
            voxel_grid_tensors = get_voxel_grid(np.array(events_chunk), height, width, num_chunks, num_bins, device)
        events_chunks.append(voxel_grid_tensors)
        if save_flag:
            if save_format == 'npy':
                for vg_index, voxel_grid_tensor in enumerate(voxel_grid_tensors):
                    save_path = os.path.join(target_folder, f'{index:06}_{vg_index:06}.npy')
                    if voxel_grid_tensor.is_cuda:
                        voxel_grid_tensor.cpu()
                    np.save(save_path, voxel_grid_tensor.cpu().numpy())
                    # print(f'save {index:06}_{vg_index:06}.pt to target_folder')
            elif save_format == 'npz':

                voxel_grid_tensors_between = {}
                for vg_index, voxel_grid_tensor in enumerate(voxel_grid_tensors):
                    if voxel_grid_tensor.is_cuda:
                        voxel_grid_tensor.cpu()
                    voxel_grid_tensors_between[f'{vg_index:06}'] = (voxel_grid_tensor.cpu().numpy())
                    # print(f'Precessed events: {index:06}_{vg_index:06}')
                save_path = os.path.join(target_folder, f'{index:06}_{j}.npz')
                np.savez_compressed(save_path, **voxel_grid_tensors_between)
                # processed_events[key] = np.array(voxel_grid_tensors_between)
            else:
                for vg_index, voxel_grid_tensor in enumerate(voxel_grid_tensors):
                    save_path = os.path.join(target_folder, f'{index:06}_{vg_index:06}.pt')
                    torch.save(voxel_grid_tensor, save_path)
                    # print(f'save {index:06}_{vg_index:06}.pt to target_folder')

    # if save_flag and save_format == 'npz':
    #     target_file = os.path.join(target_folder, 'all_processed_events.npz')
    #     print('Saving the npz file...')
    #     np.savez_compressed(target_file, **processed_events)
    #     print(f"All processed events saved to {target_file}")

    return events_chunks


if __name__ == '__main__':
    # image_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Left Images/000000.png'
    # exposure_time = 0.034  # 曝光时间，根据需要调整
    # gamma_value = 2.2  # 伽马值，根据CRF调整
    #
    # image = read_img_0_to_1(image_path)
    # # 处理图像
    # hdr_image = apply_inverse_crf_and_normalize(image, exposure_time, gamma_value)
    # result_image = transfer_hdr_to_ldr(hdr_image, exposure_time / 3, gamma_value)
    # print(result_image)
    #
    # # 保存或显示结果
    # cv2.imshow('Processed Image', result_image)
    # print('press \'q\' or \'Esc\' to quit')
    # k = cv2.waitKey(0)
    # if k == 27 or k == ord('q'):  # 按下 ESC(27) 或 'q' 退出
    #     cv2.destroyAllWindows()

    # # process image and save to a path
    # image_folder = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_d/Interlaken Left Images'
    # output_folder = '/Volumes/CenJim/train data/dataset/DSEC/train_sequences/sequence_0000001/ldr_images'
    # supervised_folder = '/Volumes/CenJim/train data/dataset/DSEC/train_sequences/sequence_0000001/hdr_images'
    # image_timestamps_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_d/Interlaken Image Exposure Left.txt'
    # process_images(image_folder, output_folder, supervised_folder, image_timestamps_path, 'npy')

    # process events and save to a path
    event_folder = '/home/s2491540/dataset/DSEC/train/interlaken_00_d/Interlaken_events_left'
    output_folder = '/home/s2491540/dataset/DSEC/train_sequences/sequence_0000001/events'
    image_timestamps_path = '/home/s2491540/dataset/DSEC/train/interlaken_00_d/Interlaken_Image_Exposure_Left.txt'
    # event_folder = '/Volumes/CenJim/train data/dataset/DSEC/test/thun_01_a/DSEC Events Left'
    # output_folder = '/Volumes/CenJim/train data/dataset/DSEC/test/thun_01_a/DSEC Events Left processed'
    # image_timestamps_path = '/Volumes/CenJim/train data/dataset/DSEC/test/thun_01_a/Thun 01 A Image Exposure Left.txt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_events(event_folder, output_folder, image_timestamps_path, 640, 480, 8, 5, device, True, 'npz')
