"""
@time    : 2022/8/17 21:22
@author  : x1aolata
@file    : Iron3D.py
@script  : 纯铁晶粒 3D数据集
            训练集: 前 148 层
            测试集: 后 148 层
"""
import sys
sys.path.append('/root/data/Segmentation')
import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import random
from segmentation3D.utils.util import get_random_crop

random_seed = 2023


class Iron3D(torch.utils.data.Dataset):
    """
    纯铁晶粒 3D数据集
    """

    def __init__(self, is_train=True, image_layers=16, image_size=256, root_path=None, train_label_rate=0.05,
                 random_seed=2023):
        super(Iron3D, self).__init__()

        self.is_train = is_train
        self.random_seed = random_seed
        if root_path is None:
            # self.root_path = '/jiangruohui/_datasets/iron/'  # 浪潮
            self.root_path = '/root/data/_datasets/iron/'  # 本部思腾
        else:
            self.root_path = root_path

        self._norm_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 裁剪尺寸
        self.image_layers = image_layers  # 层数 默认16
        self.image_size = image_size  # 图像尺寸image_size*image_size 默认256

        # 读取数据，进行训练集数据集划分

        self.images = np.load(self.root_path + 'image.npy')
        # self.labels = np.load(self.root_path + 'boundary.npy')
        self.labels = np.load(self.root_path + 'boundary_dilate1.npy')  # 使用膨胀像素的版本

        if is_train:  # 训练集 取前148层
            self.images = self.images[:, :, 0:148]
            self.labels = self.labels[:, :, 0:148]
        else:  # 测试集 取后148层
            self.images = self.images[:, :, 148:]
            self.labels = self.labels[:, :, 148:]

        # print(self.images.shape)
        # print(self.labels.shape)

        # 随机确定ban位
        if is_train:
            n_images = self.images.shape[2]
            self.is_labels = np.zeros(n_images, dtype=np.int)
            arr = list(range(n_images))
            np.random.seed(self.random_seed)
            np.random.shuffle(arr)
            self.is_labels[arr[:int(n_images * train_label_rate)]] = 1
            print(self.is_labels)
            # print('ban')
            print(self.is_labels.shape)
            print(list(self.is_labels))
            """
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]"""

    def __getitem__(self, idx):
        # 读取图像
        if self.is_train:  # 训练时 随机读取图像
            crops_index = get_random_crop(self.images.shape, (self.image_size, self.image_size, self.image_layers))
            # print(crops_index)
            crop_x, crop_y, crop_z = crops_index
            """ban 强制限制模型只能取偶数"""
            # while crop_z[0] % 2 != 0:
            #     crops_index = get_random_crop(self.images.shape, (self.image_size, self.image_size, self.image_layers))
            #     # print(crops_index)
            #     crop_x, crop_y, crop_z = crops_index
            # print(crop_z)
            # print(crop_x, crop_y, crop_z)  # (482, 738) (267, 523) (27, 43)
        else:  # 测试时 顺序读取图像
            # xy
            base = 256
            crop_x, crop_y = (base, base + self.image_size), (base, base + self.image_size)
            crop_z = (idx * self.image_layers, idx * self.image_layers + self.image_layers)

        # print(crop_x, crop_y, crop_z, sep=';')

        image = self.images[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
        label = self.labels[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
        # 将 0-255 缩限到 0-1
        label = np.clip(label, 0, 1)

        # print('image.shape', image.shape)
        # print('label.shape', label.shape)

        # print(np.unique(image))
        # print(np.unique(label))

        # image = np.transpose(image, (2, 0, 1))
        # label = np.transpose(label, (2, 0, 1))
        # print(image.shape)
        image = self._norm_transform(image / 255).type(torch.FloatTensor)
        label = self._norm_transform(label).type(torch.LongTensor)
        # print(label)
        # print('image.shape', image.shape)
        # print('label.shape', label.shape)
        # c d h w
        if self.is_train:
            return image.unsqueeze(0), label.unsqueeze(0), self.is_labels[crop_z[0]:crop_z[1]]
        else:
            return image.unsqueeze(0), label.unsqueeze(0), 1

    def __len__(self):
        if self.is_train:
            return 64
        else:
            return 8


if __name__ == '__main__':
    # data = Iron3D(is_train=True, image_layers=16, image_size=256)
    # print(data[0][0].shape)
    # print(data[0][1].shape)
    # # # print(data[0][0][0, 0, :])
    # # # print(data[0][1][0, :])
    # # # print(data[0][0][0, -1, :])
    # # # print(data[0][1][-1, :])
    # print(data[0][0].dtype)
    # print(data[0][1].dtype)
    #
    # # print(torch.unique(data[0][0]))
    # # print(torch.unique(data[0][1]))

    data = Iron3D(is_train=True, image_layers=16, image_size=256)
    # print(data.__len__())
    data_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, num_workers=8)  # 测试batch_size g改为1

    for data in data_loader:
        feature, label, is_labeled = data
        # print(feature)
        print(feature.shape)
        print(label.shape)
        print(is_labeled)

    # print(data[0][0].shape)
    # print(data[0][1].shape)
    # print(data[0][2])
    # # # print(data[0][0][0, 0, :])
    # # # print(data[0][1][0, :])
    # # # print(data[0][0][0, -1, :])
    # # # print(data[0][1][-1, :])
    # print(data[0][0].dtype)
    # print(data[0][1].dtype)
    #
    # # print(torch.unique(data[0][0]))
    # # print(torch.unique(data[0][1]))
