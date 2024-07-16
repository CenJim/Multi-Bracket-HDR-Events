import argparse
from model.network import EHDR_network
from PIL import Image
import numpy as np
import torch
import os
import torch.nn as nn


def data_load(group, device):
    # 读取数据
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
    ldr_image_1_tensor = torch.from_numpy(ldr_image_1[:, :468, :]).float().unsqueeze(0).to(device)
    ldr_image_2_tensor = torch.from_numpy(ldr_image_2[:, :468, :]).float().unsqueeze(0).to(device)
    ldr_image_3_tensor = torch.from_numpy(ldr_image_3[:, :468, :]).float().unsqueeze(0).to(device)
    events_1_tensor = torch.from_numpy(np.array(events_1)[:, :, :468, :]).float().unsqueeze(0).to(device)
    events_2_tensor = torch.from_numpy(np.array(events_2)[:, :, :468, :]).float().unsqueeze(0).to(device)

    return ldr_image_1_tensor, ldr_image_2_tensor, ldr_image_3_tensor, events_1_tensor, events_2_tensor


def main(model_name: str, pretrain_models: str, input_path: str, save_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = eval(model_name)(event_shape=(469, 640), num_feat=64, num_frame=3)
    net.eval()
    net = nn.DataParallel(net).to(device)
    state_dict = torch.load(pretrain_models, device)
    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    os.path.join(input_path, 'ldr_images/000001_2.npy')
    reference_image = os.path.join(input_path, 'ldr_images/000001_2.npy')
    under_exposure = os.path.join(input_path, 'ldr_images/000000_1.npy')
    over_exposure = os.path.join(input_path, 'ldr_images/000002_3.npy')
    events_under = os.path.join(input_path, 'events/000000_1.npz')
    events_upper = os.path.join(input_path, 'events/000001_2.npz')
    group = (under_exposure, reference_image, over_exposure, events_under, events_upper)
    input_data = data_load(group, device)
    with torch.no_grad():
        output = net(input_data[1], input_data[0], input_data[2], input_data[3],
                     input_data[4]).cpu().detach().numpy()
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
    pretrain_models = '/home/s2491540/Pythonproj/Multi-Bracket-HDR-Events/pretrained_models/EHDR.pth'
    save_path = './result'
    input_path = '/home/s2491540/dataset/DSEC/train_sequences/sequence_0000000'
    main(model_name, pretrain_models, input_path, save_path)
