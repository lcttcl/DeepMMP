
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

        weight = self._class_weights(input)


        if is_labeled is not None:

            ban = torch.ones(input.shape).to(device)
            n = input.shape[0]

            for instance in range(n):
                for i, labeled in enumerate(is_labeled[instance]):
                    if labeled == 0:
                        ban[instance, :, i, :, :] = 0
            input = input * ban


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
        output = proc_morphology.dilate(output, 3)
        output = proc_morphology.skeletonize(output)
        output = proc_morphology.remove_small_objects(output, 30)
        output = proc_morphology.dilate(output, 1)

        proc_outputs[i, :, :] = output

    return proc_outputs


def eval(labels, outputs):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    labels = np.array(labels)
    outputs = np.array(outputs).astype(labels.dtype)

    outputs = proc_output(outputs)


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

    return res


def get_eval(data: np.ndarray):

    mean_data = []
    for i in range(data.shape[0]):
        mean_data.append(np.mean(data[i, :, :], axis=0))
    return np.mean(np.array(mean_data), axis=0)


def saveTest3D(epoch, images, labels, outputs):

    images = np.array(images)
    images.astype(np.int32)
    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])

    labels = np.array(labels)
    labels.astype(np.int32)
    labels = labels.reshape(labels.shape[0], 1, labels.shape[1], labels.shape[2])

    outputs = np.array(outputs)
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    outputs.astype(np.int32)
    outputs = outputs.reshape(outputs.shape[0], 1, outputs.shape[1], outputs.shape[2])


if __name__ == '__main__':
    get_info()

    from utils.Config import Config
    from nets.unet3d.model import UNet3D
    from nets.unet3d.model import UNet2D
    import sys

    sys.path.append('/root/data/Segmentation/segmentation3D')
    sys.path.append('/root/data/Segmentation')
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-ft', '--finetune', action='store_true', default=False, help='是否预训练')
    parser.add_argument('-r', '--train_label_rate', required=True, type=float, help='train_label_rate 比率')
    parser.add_argument('-m', '--model', required=False, type=str, help='模型选择')

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

    f = open(resultfile.output_path, 'w')
    sys.stdout = f
    get_info()
    Model_Net = None
    if config['model']['name'] == 'UNet3D':
        Model_Net = UNet3D
    elif config['model']['name'] == 'UNet2D':
        Model_Net = UNet2D
    else:
        raise ValueError(f"config['model']['name'] must in ['UNet3D', 'UNet2D']")

    net = Model_Net(in_channels=config['model']['config']['in_channels'],
                    out_channels=config['model']['config']['out_channels'],
                    f_maps=config['model']['config']['f_maps'],
                    final_sigmoid=config['model']['config']['final_sigmoid'],
                    is_segmentation=config['model']['config']['is_segmentation'])




    print('remark: ', config['common']['remark'])

    train_set = Iron(phase='train',
                     image_layers=config['data']['train']['image_layers'],
                     image_size=config['data']['train']['image_size'],
                     train_label_rate=config['data']['train']['train_label_rate'],
                     random_seed=config['common']['random_seed'])

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config['data']['train']['batch_size'],
                                               shuffle=config['data']['train']['shuffle'],
                                               num_workers=8)

    test_set = Iron(phase='val',
                    image_layers=config['data']['val']['image_layers'],
                    image_size=config['data']['val']['image_size'],
                    train_label_rate=1,
                    random_seed=config['common']['random_seed'])

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config['data']['val']['batch_size'],
                                              shuffle=config['data']['val']['shuffle'],
                                              num_workers=8)

    device = torch.device("cuda:0" if config['train']['use_gpu'] and torch.cuda.is_available else "cpu")

    num_epochs = config['train']['epochs']

    opt = torch.optim.Adam(net.parameters(),
                           lr=config['train']['optimizer']['lr'],
                           weight_decay=config['train']['optimizer']['weight_decay'])

    loss = WeightedCrossEntropyLoss()


    net.to(device)


    weight_path = config['train']['fine_tune_path']
    print('weight_path:', weight_path)
    if weight_path:
        model_state_dict = torch.load(weight_path)['model_state_dict']
        keys = list(model_state_dict.keys())
        for key in keys:
            if 'encoders' not in key or 'encoders' not in key:
                del model_state_dict[key]
        net.load_state_dict(model_state_dict, strict=False)

        resultfile.saveParams(net, opt)




    for epoch in range(num_epochs):
        time_start = time.time()
        net.train()
        train_losses = RunningAverage()
        for i, data in enumerate(train_loader, 0):
            feature, label, is_labeled = data
            feature, label = feature.to(device), label.to(device)

            output = net(feature)

            l = loss(output, label.squeeze(1), is_labeled=is_labeled)

            opt.zero_grad()
            l.backward()
            opt.step()
            train_losses.update(l.item())


        time_end = time.time()
        resultfile.saveParams(net, opt)
        print((f'Training Epoch [{epoch}/{num_epochs}] loss:{train_losses.avg:.6f} time:{time_end - time_start:.2f}s'))
        writer.add_scalar(tag='train_loss', scalar_value=train_losses.avg, global_step=epoch)


        time_start = time.time()


        net.eval()
        test_losses = RunningAverage()
        eval_data = []
        for i, data in enumerate(test_loader, 0):
            feature, label, _ = data
            feature, label = feature.to(device), label.to(device)

            output = net(feature)
            eval_res = eval(label[0, 0].cpu(), output[0, 1].cpu().detach())
            eval_data.append(eval_res)
            l = loss(output, label.squeeze(1))

            test_losses.update(l.item())
        resultfile.saveTest3D(epoch, feature[0, 0].cpu(), label[0, 0].cpu(), output[0, 1].cpu().detach())
        time_end = time.time()
        saveTest3D(epoch, feature[0, 0].cpu(), label[0, 0].cpu(), output[0, 1].cpu().detach())
        print((f'Testing  Epoch [{epoch}/{num_epochs}] loss:{test_losses.avg:.6f} time:{time_end - time_start:.2f}s'))
        writer.add_scalar(tag='test_loss', scalar_value=test_losses.avg, global_step=epoch)

        pa, dice, ari, vi, map = get_eval(np.array(eval_data))
        writer.add_scalar(tag='PA', scalar_value=pa, global_step=epoch)
        writer.add_scalar(tag='Dice', scalar_value=dice, global_step=epoch)
        writer.add_scalar(tag='ARI', scalar_value=ari, global_step=epoch)
        writer.add_scalar(tag='VI', scalar_value=vi, global_step=epoch)
        writer.add_scalar(tag='mAP', scalar_value=map, global_step=epoch)


        print(f'PA: {pa:.3f},'
              f'Dice: {dice:.3f},'
              f'ARI: {ari:.3f},'
              f'vi: {vi:.3f}, '
              f'map: {map:.3f}')
        print()
