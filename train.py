import argparse
import copy
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import functional as TF
import torch.nn as nn
import lpips
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from model.network import EHDR_network


class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 包含所有sequence文件夹的根目录
        transform: torchvision.transforms 对象，用于对输出图像进行处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.groups = []

        # search for all sequence dirs
        sequences = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir)) if
                     os.path.isdir(os.path.join(root_dir, d))]

        for seq in sequences:
            ldr_image_folder = os.path.join(seq, 'ldr_images')
            events_folder = os.path.join(seq, 'events')
            hdr_image_folder = os.path.join(seq, 'hdr_images')
            print('ldr image folder: ' + ldr_image_folder)
            # 获取所有输入文件和输出文件
            ldr_image_files_1 = [f for f in sorted(os.listdir(ldr_image_folder)) if f.split('_')[1] == '1.npy']
            ldr_image_files_2 = [f for f in sorted(os.listdir(ldr_image_folder)) if f.split('_')[1] == '2.npy']
            ldr_image_files_3 = [f for f in sorted(os.listdir(ldr_image_folder)) if f.split('_')[1] == '3.npy']
            events_files_1 = [f for f in sorted(os.listdir(events_folder)) if f.split('_')[1] == '1.npz']
            events_files_2 = [f for f in sorted(os.listdir(events_folder)) if f.split('_')[1] == '2.npz']
            hdr_image_files = [f for f in sorted(os.listdir(hdr_image_folder)) if os.path.splitext(f)[1] == '.npy']

            # 确保输入和输出数量相同
            pop_num = len(ldr_image_files_3) - len(hdr_image_files)
            if pop_num < 0:
                hdr_image_files = hdr_image_files[:pop_num]
            for index, hdr_image_file in enumerate(hdr_image_files):
                self.groups.append((os.path.join(ldr_image_folder, ldr_image_files_1[index]),
                                    os.path.join(ldr_image_folder, ldr_image_files_2[index]),
                                    os.path.join(ldr_image_folder, ldr_image_files_3[index]),
                                    os.path.join(events_folder, events_files_1[index]),
                                    os.path.join(events_folder, events_files_2[index]),
                                    os.path.join(hdr_image_folder, hdr_image_file)))
        print(f'length of the groups: {len(self.groups)}')

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # get the index of the layer of the voxel grid
        # layer_index = int(idx % 5)
        # file_index = int(np.floor(idx / 5))

        # get the file path of the voxel grid
        group = self.groups[idx]

        # 读取数据
        ldr_image_1 = np.load(group[0])
        ldr_image_2 = np.load(group[1])
        ldr_image_3 = np.load(group[2])
        events_1 = []
        events_2 = []
        hdr_image = np.load(group[5])
        with np.load(group[3]) as data:
            for key in data:
                events_1.append(data[key])
        with np.load(group[4]) as data:
            for key in data:
                events_2.append(data[key])

        # 转换输入数据为torch.Tensor
        ldr_image_1_tensor = torch.from_numpy(ldr_image_1).float()
        ldr_image_2_tensor = torch.from_numpy(ldr_image_2).float()
        ldr_image_3_tensor = torch.from_numpy(ldr_image_3).float()
        events_1_tensor = torch.from_numpy(np.array(events_1)).float()
        events_2_tensor = torch.from_numpy(np.array(events_2)).float()
        hdr_image_tensor = torch.from_numpy(hdr_image).float()

        tensors_list = [ldr_image_1_tensor, ldr_image_2_tensor, ldr_image_3_tensor, events_1_tensor, events_2_tensor,
                        hdr_image_tensor]
        # 如果有传入转换器，则应用转换
        if self.transform:
            tensors_list = self.transform(tensors_list)

        return tensors_list[0], tensors_list[1], tensors_list[2], tensors_list[3], tensors_list[4], tensors_list[5]


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        lpips_value = torch.mean(self.lpips_loss(pred, target))
        l1_value = self.l1_loss(pred, target)
        # print(f'lpips_value: {lpips_value}')
        # print(f'l1_value: {l1_value}')
        return lpips_value + l1_value


class RandomRotate90:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)  # 随机选择一个角度
        return transforms.functional.rotate(x, angle)  # 应用旋转


class RandomTransform:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
            RandomRotate90(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.Lambda(lambda x: x[torch.randperm(3)])
        ])

    def __call__(self, data_list: list):
        """
        :param data_list: ldr_image_1,2,3 events_1,2 hdr_image_target
        :return transformed_data_list: ldr_image_1,2,3 events_1,2 hdr_image_target
        """
        # 设置随机种子以确保图片和标签使用相同的变换
        seed = np.random.randint(2147483647)

        transformed_data_list = []
        for data in data_list:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            transformed_data_list.append(self.transform(data))

        return transformed_data_list


def main(model_name: str, pretrain_models: str, root_files: str, save_path: str, height: int, width: int,
         num_events_per_pixel: float):
    # Parameters
    epochs = 60
    batch_size = 2
    learning_rate = 1e-4
    crop_size = 256
    time_steps = 1  # Calculate loss every 5 time steps
    # kwargs = {'event_shape': (height, width), 'num_feat': 64, 'num_frame': 3}
    loss_value = 0

    # Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EHDR_network(event_shape=(crop_size, crop_size), num_feat=64, num_frame=3).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fun = CombinedLoss()
    loss_fun.to(device)

    # Data loading and transformations

    transform = RandomTransform()
    dataset = SequenceDataset(root_files, transform=transform)  # Placeholder for your dataset class
    print(f'length of dataset: {len(dataset)}')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for i, (ldr_1, ldr_2, ldr_3, events_1, events_2, hdr) in enumerate(dataloader):
            # print(f'num of iteration: {i}')
            ldr_1, ldr_2, ldr_3, events_1, events_2, hdr = ldr_1.to(device), ldr_2.to(device), ldr_3.to(
                device), events_1.to(device), events_2.to(device), hdr.to(device)

            optimizer.zero_grad()
            output = model(ldr_2, ldr_1, ldr_3, events_1, events_2)

            if (i + 1) % time_steps == 0:
                # print(f'output: {output}')
                # print(f'hdr: {hdr}')
                loss = loss_fun(output, hdr)
                loss_value = loss
                loss.backward()
                optimizer.step()
                # model.reset_states()  # Reset states if your model has this functionality

            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss_value}")

        # 每个 epoch 结束后更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}, Current learning rate: {current_lr}')
    # 保存训练好的模型
    model_path = 'pretrained_models/EHDR.pth'
    torch.save(model.state_dict(), model_path)
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-network', type=str, default='EVSNN_LIF_final')  # EVSNN_LIF_final or PAEVSNN_LIF_AMPLIF_final
    parser.add_argument('-path_to_pretrain_models', type=str, default='./pretrained_models/EVSNN.pth')  #
    parser.add_argument('-path_to_root_files', type=str, default='./data/poster_6dof_cut.txt')
    parser.add_argument('-save_path', type=str, default='./results')
    parser.add_argument('-height', type=int, default=180)
    parser.add_argument('-width', type=int, default=240)
    parser.add_argument('-num_events_per_pixel', type=float, default=0.5)
    args = parser.parse_args()
    model_name = args.network
    pretrain_models = args.path_to_pretrain_models
    root_files = args.path_to_root_files
    save_path = args.save_path
    height = args.height
    width = args.width
    num_events_per_pixel = args.num_events_per_pixel
    main(model_name, pretrain_models, root_files, save_path, height, width, num_events_per_pixel)
