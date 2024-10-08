import argparse
from tracemalloc import Traceback

import cv2
from imageio import formats

from model.network import EHDR_network
from PIL import Image
import numpy as np
import torch
import os
import torch.nn as nn
import utils.HDR as hd
import imageio as iio


def data_load(group, device, top, bottom, left, right, hdr: bool):
    # 读取数据
    if hdr:
        ldr_image_1 = np.load(group[0])['data']
        ldr_image_2 = np.load(group[1])['data']
        ldr_image_3 = np.load(group[2])['data']
    else:
        ldr_image_1 = np.load(group[0])
        ldr_image_2 = np.load(group[1])
        ldr_image_3 = np.load(group[2])
    events_1 = []
    events_2 = []
    with np.load(group[3]) as data:
        for key in data:
            events_1.append(data[key])
    with np.load(group[4]) as data:
        for key in data:
            events_2.append(data[key])

    # 转换输入数据为torch.Tensor
    if hdr:
        ldr_image_1_tensor = torch.from_numpy(ldr_image_1[:, top:bottom, left:right]).float().unsqueeze(0).to(device)
        ldr_image_2_tensor = torch.from_numpy(ldr_image_2[:, top:bottom, left:right]).float().unsqueeze(0).to(device)
        ldr_image_3_tensor = torch.from_numpy(ldr_image_3[:, top:bottom, left:right]).float().unsqueeze(0).to(device)
        events_1_tensor = torch.from_numpy(np.array(events_1)[:, :, top:bottom, left:right]).float().unsqueeze(0).to(
            device)
        events_2_tensor = torch.from_numpy(np.array(events_2)[:, :, top:bottom, left:right]).float().unsqueeze(0).to(
            device)
    else:
        ldr_image_1_tensor = torch.from_numpy(ldr_image_1[:, :468, :]).float().unsqueeze(0).to(device)
        ldr_image_2_tensor = torch.from_numpy(ldr_image_2[:, :468, :]).float().unsqueeze(0).to(device)
        ldr_image_3_tensor = torch.from_numpy(ldr_image_3[:, :468, :]).float().unsqueeze(0).to(device)
        events_1_tensor = torch.from_numpy(np.array(events_1)[:, :, :468, :]).float().unsqueeze(0).to(device)
        events_2_tensor = torch.from_numpy(np.array(events_2)[:, :, :468, :]).float().unsqueeze(0).to(device)

    return ldr_image_1_tensor, ldr_image_2_tensor, ldr_image_3_tensor, events_1_tensor, events_2_tensor


def inference(model_name: str, pretrain_models: str, input_data, save_path: str = None, hdr: bool = False,
              compress: str = 'PQ', save_flag: bool = True, suffix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # top = 230
    # bottom = 830
    # left = 650
    # right = 1250
    top = 0
    bottom = 600
    left = 0
    right = 960
    if hdr:
        net = eval(model_name)(event_shape=(bottom - top, right - left), num_feat=64, num_frame=3)
    else:
        net = eval(model_name)(event_shape=(468, 640), num_feat=64, num_frame=3)
    net.eval()
    state_dict = torch.load(pretrain_models)
    # new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net = nn.DataParallel(net)
    net = net.to(device)
    net.load_state_dict(state_dict)

    with torch.no_grad():
        output = net(input_data[1], input_data[0], input_data[2], input_data[3],
                     input_data[4]).cpu().detach().numpy().astype(np.float32)
        output = np.transpose(output[0], (1, 2, 0))
    if hdr:
        u = 5000
        output = (((1 + u) ** output) / u).astype(np.float32)
        if compress == 'gamma':
            output = hd.pq_2_linear(output).astype(np.float32)
            # output = hd.rec2020_2_sRGB(output).astype(np.float64)
            exposure_time = hd.histogram_based_exposure(output, target_percentile=99, target_value=0.9, gamma=2.2,
                                                        tol=0.01,
                                                        max_iter=100)
            output = hd.change_exposure(output, 1).astype(np.float32)
            output = hd.apply_gamma(output, exposure_time, 2.2).astype(np.float32)
        # cv2.imwrite(os.path.join(save_path, 'test_HDR_sRGB_gamma_1_4.tif'),
        #                          cv2.cvtColor((output * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
        if save_flag:
            iio.v3.imwrite(os.path.join(save_path, f'test_HDR_Rec2020_{compress}_{suffix}.tif'),
                           (output * 65535).astype(np.uint16))
    else:
        if save_flag:
            img = Image.fromarray((output * 255).astype(np.uint8), 'RGB')
            img.save(os.path.join(save_path, 'test.bmp'))
    return output


def stitch_images(image_paths, output_path, output_size=(1900, 1060)):
    """
    将四张16位TIF格式的三通道图像按左上、右上、左下、右下顺序拼接成一张大图。

    :param image_paths: 包含四张图像路径的列表，顺序为左上、右上、左下、右下
    :param output_path: 拼接后的图像保存路径
    :param output_size: 输出图像的大小，默认为(1900, 1060)
    """
    output_width, output_height = output_size
    half_width, half_height = output_width // 2, output_height // 2

    # 读取图像并转换为 numpy 数组
    img_topleft = iio.v3.imread(image_paths[0])
    img_topright = iio.v3.imread(image_paths[1])
    img_bottomleft = iio.v3.imread(image_paths[2])
    img_bottomright = iio.v3.imread(image_paths[3])

    # 裁剪图像
    img_topleft_cropped = img_topleft[:half_height, :half_width]
    img_topright_cropped = img_topright[:half_height, -half_width:]
    img_bottomleft_cropped = img_bottomleft[-half_height:, :half_width]
    img_bottomright_cropped = img_bottomright[-half_height:, -half_width:]

    # 创建一个空的 numpy 数组用于拼接
    stitched_image = np.zeros((output_height, output_width, 3), dtype=np.uint16)

    # 拼接图像
    stitched_image[:half_height, :half_width] = img_topleft_cropped
    stitched_image[:half_height, half_width:] = img_topright_cropped
    stitched_image[half_height:, :half_width] = img_bottomleft_cropped
    stitched_image[half_height:, half_width:] = img_bottomright_cropped

    # 保存为16位三通道TIF图像
    iio.v3.imwrite(output_path, stitched_image)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-network', type=str, default='EHDR_network')  # EVSNN_LIF_final or PAEVSNN_LIF_AMPLIF_final
    # parser.add_argument('-path_to_pretrain_models', type=str, default='./pretrained_models/EVSNN.pth')  #
    # parser.add_argument('-path_to_event_files', type=str, default='./data/poster_6dof_cut.txt')
    # parser.add_argument('-save_path', type=str, default='./results')
    # parser.add_argument('-height', type=int, default=180)
    # parser.add_argument('-width', type=int, default=240)
    # parser.add_argument('-num_events_per_pixel', type=float, default=0.5)
    # args = parser.parse_args()
    # model_name = args.network
    # pretrain_models = args.path_to_pretrain_models
    # event_files = args.path_to_event_files
    # save_path = args.save_path
    # height = args.height
    # width = args.width
    # num_events_per_pixel = args.num_events_per_pixel
    # top = 230
    # bottom = 830
    # left = 650
    # right = 1250

    # top = 460
    # bottom = 1060
    # left = 940
    # right = 1900
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hdr = True
    # input_path = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/sequence'
    # os.path.join(input_path, 'ldr_images/000001_2.npy')
    # if hdr:
    #     reference_image = os.path.join(input_path, 'ldr_images/poker_fullshot_045948_4.npz')
    #     under_exposure = os.path.join(input_path, 'ldr_images/poker_fullshot_045945_1.npz')
    #     over_exposure = os.path.join(input_path, 'ldr_images/poker_fullshot_045951_7.npz')
    #     events_under = os.path.join(input_path, 'events/000000_1.npz')
    #     events_upper = os.path.join(input_path, 'events/000003_4.npz')
    # else:
    #     reference_image = os.path.join(input_path, 'ldr_images/000001_2.npy')
    #     under_exposure = os.path.join(input_path, 'ldr_images/000000_1.npy')
    #     over_exposure = os.path.join(input_path, 'ldr_images/000002_3.npy')
    #     events_under = os.path.join(input_path, 'events/000000_1.npz')
    #     events_upper = os.path.join(input_path, 'events/000001_2.npz')
    # group = (under_exposure, reference_image, over_exposure, events_under, events_upper)
    # input_data = data_load(group, device, top, bottom, left, right, hdr)
    # model_name = 'EHDR_network'
    # pretrain_models = '/home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/pretrained_models/EHDR_HDR/EHDR_model_epoch_final.pth'
    # save_path = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_out'
    # inference(model_name, pretrain_models, input_data, save_path, hdr=True, compress='PQ', suffix='2_2')

    #
    path_1 = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_out/test_HDR_Rec2020_PQ_2_2_1.tif'
    path_2 = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_out/test_HDR_Rec2020_PQ_2_2_2.tif'
    path_3 = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_out/test_HDR_Rec2020_PQ_2_2_3.tif'
    path_4 = '/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_out/test_HDR_Rec2020_PQ_2_2_4.tif'
    image_paths = [path_1, path_2, path_3, path_4]
    output_path = "/localdisk/home/s2491540/HDM_HDR/figure_sequences_03/figure_out/test_HDR_Rec2020_PQ_2_2_whole.tif"
    stitch_images(image_paths, output_path)

