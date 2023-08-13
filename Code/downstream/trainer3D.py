"""
@time    : 2022/12/5 23:07
@author  : x1aolata
@file    : trainer3D.py
@script  : 3DUnet 训练
"""
import sys

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import get_info, RunningAverage
from utils.resultFile import ResultFile
from datasets.Iron3D import Iron3D
import utils.proc_morphology as proc_morphology
from datasets.Iron import Iron


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


from torch.autograd import Variable


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target, is_labeled=None):
        # print("input.shape,target.shape")
        # print(input.shape, target.shape)  # torch.Size([2, 2, 16, 256, 256]) torch.Size([2, 16, 256, 256])
        weight = self._class_weights(input)
        # print(weight.shape)  # torch.Size([2])
        # print(weight)  # tensor([0.8700, 1.1495], device='cuda:0')
        """20230110 稀疏训练 根据is_labeled 进行删除"""
        # print(input.shape)

        if is_labeled is not None:
            # print('is_labeled is not None')
            # print(is_labeled)
            ban = torch.ones(input.shape).to(device)
            n = input.shape[0]
            # print(is_labeled)
            # print(is_labeled.shape)
            for instance in range(n):
                for i, labeled in enumerate(is_labeled[instance]):
                    if labeled == 0:
                        ban[instance, :, i, :, :] = 0
            input = input * ban
        # print('计算成功')
        # """20230103 稀疏训练"""
        # ban = torch.ones(input.shape).to(device)
        # for i in [0, 2, 4, 6, 8, 10, 12, 14]:
        #     ban[:, :, i, :, :] = 0
        # input = input * ban

        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


import utils.metrics as metrics


def proc_output(outputs):
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


def eval(labels, outputs):
    # print(labels.shape)  # torch.Size([16, 256, 256])
    # print(outputs.shape)  # torch.Size([16, 256, 256])
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    labels = np.array(labels)
    outputs = np.array(outputs).astype(labels.dtype)

    # print('checkpoint')
    # print(np.unique(labels))
    # print(np.unique(outputs))

    # print('\n\n\n')
    # print(np.unique(labels))
    # print(np.unique(outputs))
    # proc_morphology._skimageshow(labels[5, :, :])
    # proc_morphology._skimageshow(outputs[5, :, :])
    outputs = proc_output(outputs)

    # proc_morphology._skimageshow(outputs[5, :, :])
    # outputs5 = outputs[5, :, :]

    # image = dilate(image, 8)
    # image = skeletonize(image)
    # image = remove_small_objects(image, 20)
    # image = dilate(image, 3)
    # image = prun(image, 8)
    # image = remove_small_objects(image, 100)

    # input()

    # """显示，验证正确性"""
    # labels[labels==1]=255
    # outputs[outputs==1]=255
    # cv2.imwrite('label.png', labels[1,:,:])
    # cv2.imwrite('output.png', outputs[1,:,:])

    res = []
    for i in range(outputs.shape[0]):
        pred = outputs[i, :, :]
        label = labels[i, :, :]
        pa = (metrics.get_pixel_accuracy(pred, label))
        dice = (metrics.get_dice(pred, label))
        ari = (metrics.get_ari(pred, label, bg_value=1))
        vi = (metrics.get_vi(pred, label, bg_value=1)[2])
        map = (metrics.get_map_2018kdsb(pred, label, bg_value=1))
        res.append([pa, dice, ari, vi, map])
        # print([pa, dice, ari, vi, map])

    return res


def get_eval(data: np.ndarray):
    """
    对评价结果进行平均处理
    :param data:
    :return:
    """
    mean_data = []
    # print(data.shape)
    for i in range(data.shape[0]):
        # print(data[i, :, :])
        mean_data.append(np.mean(data[i, :, :], axis=0))
    # print(np.mean(np.array(mean_data), axis=0))
    return np.mean(np.array(mean_data), axis=0)


def saveTest3D(epoch, images, labels, outputs):
    """
    存 tensorboard
    :param epoch:
    :param images:
    :param labels:
    :param outputs:
    :return:
    """
    images = np.array(images)
    # images = images * 255
    images.astype(np.int32)
    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])
    # writer.add_images("test_image", images, epoch)

    labels = np.array(labels)
    # labels = labels * 255
    labels.astype(np.int32)
    labels = labels.reshape(labels.shape[0], 1, labels.shape[1], labels.shape[2])
    # writer.add_images("test_label", labels, epoch)

    outputs = np.array(outputs)
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    outputs.astype(np.int32)
    outputs = outputs.reshape(outputs.shape[0], 1, outputs.shape[1], outputs.shape[2])
    # writer.add_images("test_outputs", outputs, epoch)


if __name__ == '__main__':
    get_info()

    from utils.Config import Config
    from nets.unet3d.model import UNet3D
    from nets.unet3d.model import UNet2D
    import sys

    sys.path.append('/root/data/Segmentation/segmentation3D')
    sys.path.append('/root/data/Segmentation')
    # 读取配置文件
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('-ft', '--finetune', action='store_true', default=False, help='是否预训练')
    parser.add_argument('-r', '--train_label_rate', required=True, type=float, help='train_label_rate 比率')
    parser.add_argument('-m', '--model', required=False, type=str, help='模型选择')

    # 解析命令行参数
    args = parser.parse_args()
    print(args)
    if args.finetune:
        if args.model == 'gen':
            config = Config(r'/root/data/Segmentation/segmentation3D/config/config3d_finetune_gen.yml')
        elif args.model == 'jus':
            config = Config(r'/root/data/Segmentation/segmentation3D/config/config3d_finetune_jus.yml')
        elif args.model == 'loc':
            config = Config(r'/root/data/Segmentation/segmentation3D/config/config3d_finetune_loc.yml')
        elif args.model == 'jus_loc':
            config = Config(r'/root/data/Segmentation/segmentation3D/config/config3d_finetune_jus_loc.yml')
        elif args.model == 'jus_loc_gen':
            config = Config(r'/root/data/Segmentation/segmentation3D/config/config3d_finetune_jus_loc_gen.yml')
        else:
            print('error : ')
            exit(111)
    else:
        config = Config(r'/root/data/Segmentation/segmentation3D/config/config3d.yml')

    config['data']['train']['train_label_rate'] = args.train_label_rate
    config['common']['random_seed'] = 20230529

    flag_name = config['common']['remark'] + 'train_label_rate' + str(config['data']['train']['train_label_rate'])
    resultfile = ResultFile(r'/root/data/_results', 'Segmentation3D', flag_name)

    config.save(resultfile.config_path)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(comment=flag_name)

    # 配置输出文件
    f = open(resultfile.output_path, 'w')
    sys.stdout = f
    get_info()
    # 配置网络模型
    Model_Net = None
    if config['model']['name'] == 'UNet3D':
        Model_Net = UNet3D
    elif config['model']['name'] == 'UNet2D':
        Model_Net = UNet2D
    else:
        raise ValueError(f"config['model']['name'] must in ['UNet3D', 'UNet2D']")

    # 配置网络结构
    net = Model_Net(in_channels=config['model']['config']['in_channels'],
                    out_channels=config['model']['config']['out_channels'],
                    f_maps=config['model']['config']['f_maps'],
                    final_sigmoid=config['model']['config']['final_sigmoid'],
                    is_segmentation=config['model']['config']['is_segmentation'])




    print('remark: ', config['common']['remark'])

    # 纯铁晶粒数据集配置
    # train_set = Iron3D(is_train=True,
    #                    image_layers=config['data']['train']['image_layers'],
    #                    image_size=config['data']['train']['image_size'])
    train_set = Iron(phase='train',
                     image_layers=config['data']['train']['image_layers'],
                     image_size=config['data']['train']['image_size'],
                     train_label_rate=config['data']['train']['train_label_rate'],
                     random_seed=config['common']['random_seed'])

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config['data']['train']['batch_size'],
                                               shuffle=config['data']['train']['shuffle'],
                                               num_workers=8)

    # test_set = Iron3D(is_train=False,
    #                   image_layers=config['data']['val']['image_layers'],
    #                   image_size=config['data']['val']['image_size'])
    test_set = Iron(phase='val',
                    image_layers=config['data']['val']['image_layers'],
                    image_size=config['data']['val']['image_size'],
                    train_label_rate=1,
                    random_seed=config['common']['random_seed'])

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config['data']['val']['batch_size'],
                                              shuffle=config['data']['val']['shuffle'],
                                              num_workers=8)  # 测试batch_size g改为1

    device = torch.device("cuda:0" if config['train']['use_gpu'] and torch.cuda.is_available else "cpu")

    num_epochs = config['train']['epochs']

    opt = torch.optim.Adam(net.parameters(),
                           lr=config['train']['optimizer']['lr'],
                           weight_decay=config['train']['optimizer']['weight_decay'])

    loss = WeightedCrossEntropyLoss()

    # resultfile = ResultFile(r'/jiangruohui/_results', 'Segmentation3D', 'no-fine-tune-test')

    net.to(device)

    """
    加载预训练参数fine-tune
    """
    weight_path = config['train']['fine_tune_path']
    print('weight_path:', weight_path)
    if weight_path:
        model_state_dict = torch.load(weight_path)['model_state_dict']
        keys = list(model_state_dict.keys())
        for key in keys:
            if 'encoders' not in key or 'encoders' not in key:
                del model_state_dict[key]
        print('开始加载预训练参数')
        net.load_state_dict(model_state_dict, strict=False)
        print('预训练参数加载完毕')
        resultfile.saveParams(net, opt)



    # writer.add_graph(model = net,input_to_model = torch.ones((4,1,16,256,256)))
    # print(111)

    """
    训练
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        net.train()
        train_losses = RunningAverage()
        for i, data in enumerate(train_loader, 0):
            # print('iter:', i)
            feature, label, is_labeled = data
            feature, label = feature.to(device), label.to(device)

            output = net(feature)
            # print('feature.shape', feature.shape)
            # print('label.shape', label.shape)
            # print('output.shape', output.shape)

            # print(np.unique(np.array(label[0, 0].cpu())))
            # ooo = np.array(output[0, 1].cpu().detach())
            # print(np.unique(ooo))
            # ooo2 = np.array(output[0, 0 ].cpu().detach())
            # print(np.unique(ooo2))

            # 传入稀疏标注
            l = loss(output, label.squeeze(1), is_labeled=is_labeled)

            opt.zero_grad()
            l.backward()
            opt.step()
            train_losses.update(l.item())
        # 存储训练的图片
        # resultfile.saveTrain3DOut(output[0, 1].cpu().detach(), epoch, label='output')
        # resultfile.saveTrain3DImage(feature[0, 0].cpu(), epoch, label='image')
        # resultfile.saveTrain3DLabel(label[0, 0].cpu(), epoch, label='label')

        time_end = time.time()
        resultfile.saveParams(net, opt)
        print((f'Training Epoch [{epoch}/{num_epochs}] loss:{train_losses.avg:.6f} time:{time_end - time_start:.2f}s'))
        writer.add_scalar(tag='train_loss', scalar_value=train_losses.avg, global_step=epoch)

        """
        测试
        """
        time_start = time.time()
        # 切换到测试模式

        net.eval()
        # 记录测试损失
        test_losses = RunningAverage()
        eval_data = []
        for i, data in enumerate(test_loader, 0):
            # print('iter:', i)
            feature, label, _ = data
            feature, label = feature.to(device), label.to(device)

            output = net(feature)
            # print(label.shape,output.shape) torch.Size([1, 1, 16, 256, 256]) torch.Size([1, 2, 16, 256, 256])
            eval_res = eval(label[0, 0].cpu(), output[0, 1].cpu().detach())
            eval_data.append(eval_res)
            l = loss(output, label.squeeze(1))

            test_losses.update(l.item())
        resultfile.saveTest3D(epoch, feature[0, 0].cpu(), label[0, 0].cpu(), output[0, 1].cpu().detach())
        time_end = time.time()
        # print(eval_data)
        # tensorboard 写入图像
        saveTest3D(epoch, feature[0, 0].cpu(), label[0, 0].cpu(), output[0, 1].cpu().detach())
        print((f'Testing  Epoch [{epoch}/{num_epochs}] loss:{test_losses.avg:.6f} time:{time_end - time_start:.2f}s'))
        writer.add_scalar(tag='test_loss', scalar_value=test_losses.avg, global_step=epoch)

        # print(get_eval(np.array(eval_data)))
        pa, dice, ari, vi, map = get_eval(np.array(eval_data))
        writer.add_scalar(tag='PA', scalar_value=pa, global_step=epoch)
        writer.add_scalar(tag='Dice', scalar_value=dice, global_step=epoch)
        writer.add_scalar(tag='ARI', scalar_value=ari, global_step=epoch)
        writer.add_scalar(tag='VI', scalar_value=vi, global_step=epoch)
        writer.add_scalar(tag='mAP', scalar_value=map, global_step=epoch)

        # writer.add_images("test_image", img, epoch)
        # writer.add_images("test_image", img, epoch)
        # writer.add_images("test_image", img, epoch)

        print(f'PA: {pa:.3f},'
              f'Dice: {dice:.3f},'
              f'ARI: {ari:.3f},'
              f'vi: {vi:.3f}, '
              f'map: {map:.3f}')
        print()
