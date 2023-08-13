
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image
import os
import torch.nn.functional as F

from datasets.Iron3D import Iron3D
from nets.Unet3D import UNet3D
from utils.util import print2txt

filename = '[20220930]Iron_3DUnet'
weight_path = f'/root/data/Result/{filename}/unet.params'
save_path = f'/root/data/Result/{filename}/'
save_path_image = f'/root/data/Result/{filename}/image/'
save_path_image_test = f'/root/data/Result/{filename}/image_test/'

Path(save_path).mkdir(parents=True, exist_ok=True)
Path(save_path_image).mkdir(parents=True, exist_ok=True)
Path(save_path_image_test).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print2txt(device)
    print2txt(torch.cuda.get_device_name(0))

    batch_size = 1
    num_epochs = 200
    lr = 1e-4
    wd = 1e-3
    train_set = Iron3D(is_train=True)
    test_set = Iron3D(is_train=False)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    print2txt('datasets load finish!')
    net = UNet3D(in_dim=1, out_dim=2, num_filters=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.BCEWithLogitsLoss()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        time_start = time.time()
        net.train()
        running_loss = []
        for i, data in enumerate(train_iter, 0):
            feature, label = data
            feature, label = feature.to(device), label.to(device)
            output = net(feature)
            output_ = torch.softmax(output, dim=1)
            l = loss(output_, label.squeeze(1))
            opt.zero_grad()
            l.backward()
            opt.step()
            running_loss.append(l.item())

        time_end = time.time()
        print2txt(
            f'Epoch: {epoch + 1:03} loss: {np.mean(running_loss):.4f} time: {time_end - time_start:.2f}s')

        if True:
            # 保存训练图像
            output = torch.softmax(output, dim=1)
            for i in range(32):
                _image, _label = feature[0, 0, i:i + 1, :, :], label[0, 0, i:i + 1, :, :]
                _output = output[0, 0, i:i + 1, :, :]
                _output[_output >= 0.5] = 1
                _output[_output < 0.5] = 0
                img = torch.stack([_image, _output, _label], dim=0)
                save_image(img, f'{save_path_image}{epoch + 1:06}_{i:02}.png')
