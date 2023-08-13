
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


class Iron(torch.utils.data.Dataset):


    def __init__(self,
                 phase='train',
                 image_layers=16,
                 image_size=256,
                 root_path=None,
                 train_label_rate=0.05,
                 random_seed=2023):

        super(Iron, self).__init__()

        assert phase in ['train', 'val', 'test']
        self.phase = phase


        self.image_layers = image_layers
        self.image_size = image_size


        if root_path is None:
            self.root_path = '/root/data/_datasets/iron/'  #
        else:
            self.root_path = root_path

        self.train_label_rate = train_label_rate

        self.random_seed = random_seed


        self._norm_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        self.images = np.load(self.root_path + 'image.npy')
        self.labels = np.load(self.root_path + 'boundary_dilate1.npy')

        if self.phase == 'train':
            self.images = self.images[:, :, 0:148]
            self.labels = self.labels[:, :, 0:148]
        elif self.phase == 'val':
            self.images = self.images[:, :, 148:222]
            self.labels = self.labels[:, :, 148:222]
        elif self.phase == 'test':
            self.images = self.images[:, :, 222:]
            self.labels = self.labels[:, :, 222:]
        else:
            raise ValueError(f"phase must be a value in ['train', 'val', 'test']")

        print(f'{self.phase} datasets shape is :')
        print(self.images.shape)
        print(self.labels.shape)


        self.is_labels = None


        if self.phase == 'train':
            n_images = self.images.shape[2]
            self.is_labels = np.zeros(n_images, dtype=np.int32)
            arr = list(range(n_images))
            np.random.seed(self.random_seed)
            np.random.shuffle(arr)
            self.is_labels[arr[:int(n_images * self.train_label_rate)]] = 1
            for index, flag in enumerate(self.is_labels):
                if not flag:
                    self.labels[:, :, index] = 0
        else:
            n_images = self.images.shape[2]
            self.is_labels = np.ones(n_images, dtype=np.int32)
        print(f'is_labels is :\n{self.is_labels}')

    def __getitem__(self, idx):
        if self.phase == 'train':
            while True:
                crops_index = self.get_random_crop(self.images.shape,
                                                   (self.image_size, self.image_size, self.image_layers))
                crop_x, crop_y, crop_z = crops_index
                if self.is_covered(crop_z):
                    break

        elif self.phase == 'val' or self.phase == 'test':
            base = 256
            crop_x, crop_y = (base, base + self.image_size), (base, base + self.image_size)
            crop_z = (idx * self.image_layers, idx * self.image_layers + self.image_layers)
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
        if self.phase == 'train':
            return 64
        elif self.phase == 'val':
            return 4
        elif self.phase == 'test':
            return 4
        else:
            raise ValueError(f"phase must be a value in ['train', 'val', 'test']")

    def get_random_crop(self, shape: tuple, crop_size: tuple, random_seed=None):
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
        wheres = np.where(self.is_labels == 1)[0]
        for item in wheres:
            if crop_z[0] <= item < crop_z[1]:
                return True
        return False


if __name__ == '__main__':
    data = Iron(phase='val', image_layers=1, image_size=256, train_label_rate=0.1, random_seed=2023)
    print(data.__len__())

    data_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, num_workers=8)

    for data in data_loader:
        feature, label, is_labeled = data
        print(feature.shape)
        print(label.shape)
        print(is_labeled)
