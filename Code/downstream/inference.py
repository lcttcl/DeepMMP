"""
@time    : 2023/6/22 19:52
@author  : x1aolata
@file    : inference.py
@script  : ...
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import get_info, RunningAverage
from datasets.Iron3D import Iron3D
import utils.proc_morphology as proc_morphology
from datasets.IronInfer import IronInfer
import cv2
from pathlib import Path
import torchvision.transforms as transforms


class ResultFile():
    """
    管理每一次训练的中间结果保存
    """

    def __init__(self, root_path):
        self.root_path = Path(root_path)

    def save3DImage(self, images, tag_name, save_path: Path, label='image',start_index = 0):
        """
        将训练或测试的图像存入本地
        图像需要 乘 255 以回复灰度图

        :param images: [D,H,W] 如果数据在GPU中需手动转到CPU上
        :param tag_name: index标号
        :param save_path: 存储路径
        :param label: 标签
        :return:
        """
        images = np.array(images)
        base_name = f'{label}_{tag_name}'
        for i in range(images.shape[0]):
            image = images[i, :, :] * 255
            image.astype(np.int)
            save_image_path = save_path.joinpath(f'{base_name}_{i+start_index:03}.png')
            cv2.imwrite(str(save_image_path), image)

    def save3DLabel(self, labels, tag_name, save_path: Path, label='label',start_index = 0):
        """
        将训练或测试的标签存入本地
        训练标签二分类需要 转为到0,255乘 255
        :param labels: [D,H,W] 如果数据在GPU中需手动转到CPU上
        :param index:  index标号
        :param label:  标签
        :return:
        """
        self.save3DImage(labels, tag_name, save_path, label,start_index)

    def save3DOut(self, outputs, epoch, save_path: Path, label='output',start_index = 0):
        """
        将训练或推理的输出结果存入本地
        训练标签二分类需要 转为到0,255乘 255
        :param outputs: [D,H,W] 如果数据在GPU中需手动转到CPU上
        :param epoch:  index标号
        :param label:  标签
        :return:
        """
        outputs = np.array(outputs)
        base_name = f'{label}_{epoch}'
        for i in range(outputs.shape[0]):
            image = outputs[i, :, :]
            image[image >= 0.5] = 255
            image[image < 0.5] = 0
            image.astype(np.int)
            save_image_path = save_path.joinpath(f'{base_name}_{i+start_index:03}.png')
            cv2.imwrite(str(save_image_path), image)

    def saveTest3D(self, tag_name, images, labels, outputs, proc_outputs,start_index = 0):
        tag_path = self.root_path.joinpath(tag_name)
        tag_path.mkdir(parents=True, exist_ok=True)
        self.save3DImage(images, tag_name, save_path=tag_path, label='image',start_index = start_index)
        self.save3DLabel(labels, tag_name, save_path=tag_path, label='label',start_index = start_index)
        self.save3DOut(outputs, tag_name, save_path=tag_path, label='output',start_index = start_index)
        self.save3DOut(proc_outputs, tag_name, save_path=tag_path, label='proc_output',start_index = start_index)


def proc_output(outputs):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    proc_outputs = outputs.copy()

    for i in range(outputs.shape[0]):
        output = outputs[i, :, :]
        output = proc_morphology.dilate(output, 3)  # 膨胀3
        # proc_morphology._skimageshow(output)
        output = proc_morphology.skeletonize(output)  # 骨架化
        # proc_morphology._skimageshow(output)
        output = proc_morphology.remove_small_objects(output, 30)  # 去掉小目标
        # proc_morphology._skimageshow(output)
        output = proc_morphology.dilate(output, 1)  # 膨胀1
        # proc_morphology._skimageshow(output)

        proc_outputs[i, :, :] = output

    return proc_outputs


if __name__ == '__main__':
    get_info()

    from utils.Config import Config
    from nets.unet3d.model import UNet3D

    from pathlib import Path

    source_path = r'/root/data/_results/j,m bnSegmentation3D-[2023-04-03-16:12:56]-[unet3dtrain_label_rate1.0]'
    save_path = r'/root/data/exp_res/inference2/unet3dtrain_label_rate1.0'
    # source_path = r'/root/data/_results/Segmentation3D-[2023-04-07-18:23:33]-[unet3d_ft_juslocgentrain_label_rate1.0]'
    # save_path = r'/root/data/exp_res/inference2/unet3d_ft_juslocgentrain_label_rate1.0'
    source_path = Path(source_path)
    save_path = Path(save_path)

    config = Config(source_path.joinpath('config.yml'))
    net = UNet3D(in_channels=config['model']['config']['in_channels'],
                 out_channels=config['model']['config']['out_channels'],
                 f_maps=config['model']['config']['f_maps'],
                 final_sigmoid=config['model']['config']['final_sigmoid'],
                 is_segmentation=config['model']['config']['is_segmentation'])
    device = torch.device("cuda:0" if config['train']['use_gpu'] and torch.cuda.is_available else "cpu")

    weight_path = source_path.joinpath('savedModel').joinpath('params.pth')
    print('weight_path:', weight_path)
    if weight_path:
        model_state_dict = torch.load(weight_path)['model_state_dict']
        print('开始加载预训练参数')
        net.load_state_dict(model_state_dict, strict=False)
        print('预训练参数加载完毕')

    net.to(device)
    net.eval()

    # test_set = IronInfer(phase='val',
    #                 image_layers=config['data']['val']['image_layers'],
    #                 image_size=config['data']['val']['image_size'],
    #                 train_label_rate=1,
    #                 random_seed=config['common']['random_seed'])
    #
    # test_loader = torch.utils.data.DataLoader(test_set,
    #                                           batch_size=config['data']['val']['batch_size'],
    #                                           shuffle=config['data']['val']['shuffle'],
    #                                           num_workers=8)  # 测试batch_size g改为1
    # res = ResultFile(save_path)
    # for i, data in enumerate(test_loader, 0):
    #     # print('iter:', i)
    #     feature, label, _ = data
    #     feature, label = feature.to(device), label.to(device)
    #
    #     output = net(feature)
    #     proc_outputs = proc_output(output[0, 1].cpu().detach().numpy())
    #     # print(label.shape,output.shape) torch.Size([1, 1, 16, 256, 256]) torch.Size([1, 2, 16, 256, 256])
    #     print(feature.shape)
    #     print(label.shape)
    #     print(output.shape)
    #     res.saveTest3D(f"{i:03}", feature[0, 0].cpu(), label[0, 0].cpu(), output[0, 1].cpu().detach(),proc_outputs)

    images = np.load(r'/root/data/_datasets/iron/image.npy')
    labels = np.load(r'/root/data/_datasets/iron/boundary_dilate1.npy')
    images = images[:, :, 148:296]
    labels = labels[:, :, 148:296]
    _norm_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    res = ResultFile(save_path)
    for i in range(16):
        id_x = i % 4
        id_y = i // 4

        for j in range(9):
            crop_x, crop_y = (id_x * 256, id_x * 256 + 256), (id_y * 256, id_y * 256 + 256)
            crop_z = (j * 16, j * 16 + 16)
            print(crop_x, crop_y, crop_z)
            image = images[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
            label = labels[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], crop_z[0]:crop_z[1]]
            # 将 0-255 缩限到 0-1
            label = np.clip(label, 0, 1)
            image = _norm_transform(image / 255).type(torch.FloatTensor)
            label = _norm_transform(label).type(torch.LongTensor)
            feature = image.unsqueeze(0)  .unsqueeze(0)
            label = label.unsqueeze(0)  .unsqueeze(0)
            feature, label = feature.to(device), label.to(device)
            output = net(feature)
            proc_outputs = proc_output(output[0, 1].cpu().detach().numpy())
            res.saveTest3D(f"{crop_x}{crop_y}{i:03}", feature[0, 0].cpu(), label[0, 0].cpu(), output[0, 1].cpu().detach(), proc_outputs,start_index=j*16)