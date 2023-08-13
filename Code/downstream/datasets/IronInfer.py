"""
@time    : 2023/3/6 15:55
@author  : x1aolata
@file    : Iron.py
@script  : 纯铁晶粒 数据集v2.0重构版本
            2D/3D
"""
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision

import cv2
import h5py

from PIL import Image
from pathlib import Path


class IronInfer(torch.utils.data.Dataset):
    """
    纯铁晶粒 2D/3D数据集v2.0重构版本
    """

    def __init__(self,
                 phase='train',
                 image_layers=16,
                 image_size=256,
                 root_path=None,
                 train_label_rate=0.05,
                 random_seed=2023):
        """
        纯铁晶粒 2D/3D数据集v2.0重构版本
        :param phase:               数据集集模式  训练 验证 测试  ['train', 'val', 'test']
        :param image_layers:        采样层数 默认16
        :param image_size:          图像尺寸image_size*image_size 默认256
        :param root_path:           数据集位置
        :param train_label_rate:    训练集裁剪比例
        :param random_seed:         随机种子 用于训练集裁剪一致
        """
        super(IronInfer, self).__init__()

        assert phase in ['train', 'val', 'test']
        self.phase = phase  # 数据集集模式  训练 验证 测试

        # 裁剪尺寸
        self.image_layers = image_layers  # 层数 默认16
        self.image_size = image_size  # 图像尺寸image_size*image_size 默认256

        # 数据集位置
        if root_path is None:
            # self.root_path = '/jiangruohui/_datasets/iron/'  # 浪潮
            self.root_path = '/root/data/_datasets/iron/'  # 本部思腾
        else:
            self.root_path = root_path
        # 训练集裁剪比例
        self.train_label_rate = train_label_rate
        # 随机种子 用于训练集裁剪一致
        self.random_seed = random_seed

        # transforms方法
        self._norm_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 读取数据，进行数据集划分
        self.images = np.load(self.root_path + 'image.npy')
        # self.labels = np.load(self.root_path + 'boundary.npy')
        self.labels = np.load(self.root_path + 'boundary_dilate1.npy')  # 使用膨胀像素的版本

        if self.phase == 'train':  # 训练集 取前148层
            self.images = self.images[:, :, 0:148]
            self.labels = self.labels[:, :, 0:148]
        elif self.phase == 'val':  # 验证集 取148-222层
            self.images = self.images[:, :, 148:296]
            self.labels = self.labels[:, :, 148:296]
        elif self.phase == 'test':  # 测试集 取222-296层
            self.images = self.images[:, :, 222:]
            self.labels = self.labels[:, :, 222:]
        else:
            raise ValueError(f"phase must be a value in ['train', 'val', 'test']")

        print(f'{self.phase} datasets shape is :')
        print(self.images.shape)
        print(self.labels.shape)

        # 训练集中 有标签label列表
        self.is_labels = None

        # 随机确定ban位
        if self.phase == 'train':  # 训练集
            n_images = self.images.shape[2]
            self.is_labels = np.zeros(n_images, dtype=np.int32)
            arr = list(range(n_images))
            np.random.seed(self.random_seed)
            np.random.shuffle(arr)
            self.is_labels[arr[:int(n_images * self.train_label_rate)]] = 1
            # 物理隔离 ，根据 self.is_labels 将self.labels 中非需要训练的样本屏蔽 置0
            for index, flag in enumerate(self.is_labels):
                if not flag:
                    self.labels[:, :, index] = 0
        else:  # 测试集或验证集置1
            n_images = self.images.shape[2]
            self.is_labels = np.ones(n_images, dtype=np.int32)
        print(f'is_labels is :\n{self.is_labels}')

    def __getitem__(self, idx):
        if self.phase == 'train':  # 训练集
            while True:  # 反复取样，要求取样到有标签的一组数据
                crops_index = self.get_random_crop(self.images.shape,
                                                   (self.image_size, self.image_size, self.image_layers))
                crop_x, crop_y, crop_z = crops_index
                if self.is_covered(crop_z):
                    break

        elif self.phase == 'val' or self.phase == 'test':  # 验证集 or 测试集
            idzz = idx // 16
            idxy = idx % 16
            id_x = idxy % 4
            id_y = idxy // 4

            base_xy = 256
            base_z = 16
            crop_x, crop_y = (id_x * self.image_size, id_x * self.image_size + self.image_size), (
            id_y * self.image_size, id_y * self.image_size + self.image_size)
            crop_z = (idzz * self.image_layers, idzz * self.image_layers + self.image_layers)

            print(crop_x, crop_y, crop_z)
        else:
            raise ValueError(f"phase must be a value in ['train', 'val', 'test']")

        image = self.images[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
        label = self.labels[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
        # 将 0-255 缩限到 0-1
        label = np.clip(label, 0, 1)
        image = self._norm_transform(image / 255).type(torch.FloatTensor)
        label = self._norm_transform(label).type(torch.LongTensor)

        return image.unsqueeze(0), label.unsqueeze(0), self.is_labels[crop_z[0]:crop_z[1]]

    def __len__(self):
        if self.phase == 'train':  # 训练集
            return 64
        elif self.phase == 'val':  # 验证集
            return 9 * 16
        elif self.phase == 'test':  # 测试集
            return 4
        else:
            raise ValueError(f"phase must be a value in ['train', 'val', 'test']")

    def get_random_crop(self, shape: tuple, crop_size: tuple, random_seed=None):
        """
        随机裁剪，产生
        :param random_seed:
        :param shape:
        :param crop_size:
        :return:
        给出最大范围，以及给出需要裁剪的尺寸，返回需要的size
        """
        if random_seed:
            random.seed(random_seed)
        if len(shape) != len(crop_size):
            return
        result = []
        for index in range(len(shape)):
            range_item = shape[index]
            crop_size_item = crop_size[index]
            res = random.randint(0, range_item - crop_size_item)
            result.append((res, res + crop_size_item))
        if random_seed:
            random.seed()

        return result

    def is_covered(self, crop_z):
        """
        判断采样后的crop_z中是否有有标签的数据
        :param crop_z:
        :return:
        """
        wheres = np.where(self.is_labels == 1)[0]
        for item in wheres:
            if crop_z[0] <= item < crop_z[1]:
                return True
        return False


if __name__ == '__main__':
    data = IronInfer(phase='val', image_layers=16, image_size=256, train_label_rate=0.1, random_seed=2023)
    print(data.__len__())
    # data = Iron(phase='val', image_layers=16, image_size=256, train_label_rate=0.05, random_seed=2023)
    # print(data.__len__())
    # data = Iron(phase='test', image_layers=16, image_size=256, train_label_rate=0.05, random_seed=2023)
    # print(data.__len__())

    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    for data in data_loader:
        feature, label, is_labeled = data
        print(feature.shape)
        print(label.shape)
        print(is_labeled)
        # feature = feature[0, 0, :, :, :]
        # label = label[0, 0, :, :, :]
        # is_labeled = is_labeled[0]
        # os.makedirs('tmpfile')
        # for i in range(feature.shape[0]):
        #     cv2.imwrite(
        #         f'/root/data/Segmentation/segmentation3D/datasets/tmpfile/frature{i}_is_labeled{is_labeled[i]}.png',
        #         np.array(feature[i, :, :]) * 255)
        #     cv2.imwrite(
        #         f'/root/data/Segmentation/segmentation3D/datasets/tmpfile/label{i}_is_labeled{is_labeled[i]}.png',
        #         np.array(label[i, :, :]) * 255)
        #
        # break
