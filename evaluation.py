import argparse
from tracemalloc import Traceback

import cv2

from model.network import EHDR_network
from PIL import Image
import numpy as np
import torch
import os
import torch.nn as nn
import utils.HDR as hd


def data_load(group, device, top ,bottom, left, right, hdr: bool):
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
        events_1_tensor = torch.from_numpy(np.array(events_1)[:, :, top:bottom, left:right]).float().unsqueeze(0).to(device)
        events_2_tensor = torch.from_numpy(np.array(events_2)[:, :, top:bottom, left:right]).float().unsqueeze(0).to(device)
    else:
        ldr_image_1_tensor = torch.from_numpy(ldr_image_1[:, :468, :]).float().unsqueeze(0).to(device)
        ldr_image_2_tensor = torch.from_numpy(ldr_image_2[:, :468, :]).float().unsqueeze(0).to(device)
        ldr_image_3_tensor = torch.from_numpy(ldr_image_3[:, :468, :]).float().unsqueeze(0).to(device)
        events_1_tensor = torch.from_numpy(np.array(events_1)[:, :, :468, :]).float().unsqueeze(0).to(device)
        events_2_tensor = torch.from_numpy(np.array(events_2)[:, :, :468, :]).float().unsqueeze(0).to(device)

    return ldr_image_1_tensor, ldr_image_2_tensor, ldr_image_3_tensor, events_1_tensor, events_2_tensor


def main(model_name: str, pretrain_models: str, input_path: str, save_path: str, hdr: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    top = 230
    bottom = 830
    left = 650
    right = 1250
    if hdr:
        net = eval(model_name)(event_shape=(bottom - top, right - left), num_feat=64, num_frame=3)
    else:
        net = eval(model_name)(event_shape=(468, 640), num_feat=64, num_frame=3)
    net.eval()
    state_dict = torch.load(pretrain_models)
    # new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    net = nn.DataParallel(net)
    net.load_state_dict(state_dict)
    net = net.to(device)

    os.path.join(input_path, 'ldr_images/000001_2.npy')
    if hdr:
        reference_image = os.path.join(input_path, 'ldr_images/showgirl_02_301963_1.npz')
        under_exposure = os.path.join(input_path, 'ldr_images/showgirl_02_301966_4.npz')
        over_exposure = os.path.join(input_path, 'ldr_images/showgirl_02_301969_7.npz')
        events_under = os.path.join(input_path, 'events/000000_1.npz')
        events_upper = os.path.join(input_path, 'events/000003_4.npz')
    else:
        reference_image = os.path.join(input_path, 'ldr_images/000001_2.npy')
        under_exposure = os.path.join(input_path, 'ldr_images/000000_1.npy')
        over_exposure = os.path.join(input_path, 'ldr_images/000002_3.npy')
        events_under = os.path.join(input_path, 'events/000000_1.npz')
        events_upper = os.path.join(input_path, 'events/000001_2.npz')
    group = (under_exposure, reference_image, over_exposure, events_under, events_upper)
    input_data = data_load(group, device, top ,bottom, left, right, hdr)
    with torch.no_grad():
        output = net(input_data[1], input_data[0], input_data[2], input_data[3],
                     input_data[4]).cpu().detach().numpy().astype(np.float64)
        output = np.transpose(output[0], (1, 2, 0))
    if hdr:
        u = 5000
        output = ((1 + u) ** output) / u
        output = hd.pq_2_linear(output)
        # output = hd.rec2020_2_sRGB(output)
        exposure_time = hd.histogram_based_exposure(output, target_percentile=99, target_value=0.9, gamma=2.2, tol=0.01,
                                                     max_iter=100)
        output = hd.change_exposure(output, 1)
        output = hd.apply_gamma(output, exposure_time, 2.2)
        cv2.imwrite(os.path.join(save_path, 'test_HDR_rec2020_gamma_1_3.tif'),
                                 cv2.cvtColor((output * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
    else:
        output = (output * 255).astype(np.uint8)
        img = Image.fromarray(output, 'RGB')
        img.save(os.path.join(save_path, 'test.bmp'))


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
    model_name = 'EHDR_network'
    pretrain_models = '/home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/pretrained_models/1.3-fishing_longshot/EHDR_model_epoch_final.pth'
    save_path = '/home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/result'
    input_path = '/home/s2491540/dataset/HDM_HDR/sequences_not_for_train/showgirl_02'
    main(model_name, pretrain_models, input_path, save_path, hdr=True)
