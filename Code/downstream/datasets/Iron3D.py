
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


    def __init__(self, is_train=True, image_layers=16, image_size=256, root_path=None, train_label_rate=0.05,
                 random_seed=2023):
        super(Iron3D, self).__init__()

        self.is_train = is_train
        self.random_seed = random_seed
        if root_path is None:
            self.root_path = '/root/data/_datasets/iron/'
        else:
            self.root_path = root_path

        self._norm_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        self.image_layers = image_layers
        self.image_size = image_size



        self.images = np.load(self.root_path + 'image.npy')

        self.labels = np.load(self.root_path + 'boundary_dilate1.npy')

        if is_train:
            self.images = self.images[:, :, 0:148]
            self.labels = self.labels[:, :, 0:148]
        else:
            self.images = self.images[:, :, 148:]
            self.labels = self.labels[:, :, 148:]


        if is_train:
            n_images = self.images.shape[2]
            self.is_labels = np.zeros(n_images, dtype=np.int)
            arr = list(range(n_images))
            np.random.seed(self.random_seed)
            np.random.shuffle(arr)
            self.is_labels[arr[:int(n_images * train_label_rate)]] = 1
            print(self.is_labels)

            print(self.is_labels.shape)
            print(list(self.is_labels))
            """
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]"""

    def __getitem__(self, idx):
        if self.is_train:
            crops_index = get_random_crop(self.images.shape, (self.image_size, self.image_size, self.image_layers))

            crop_x, crop_y, crop_z = crops_index

        else:
            # xy
            base = 256
            crop_x, crop_y = (base, base + self.image_size), (base, base + self.image_size)
            crop_z = (idx * self.image_layers, idx * self.image_layers + self.image_layers)


        image = self.images[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
        label = self.labels[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]

        label = np.clip(label, 0, 1)

        image = self._norm_transform(image / 255).type(torch.FloatTensor)
        label = self._norm_transform(label).type(torch.LongTensor)

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
    data = Iron3D(is_train=True, image_layers=16, image_size=256)
    # print(data.__len__())
    data_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, num_workers=8)  # 测试batch_size g改为1

    for data in data_loader:
        feature, label, is_labeled = data
        print(feature.shape)
        print(label.shape)
        print(is_labeled)


