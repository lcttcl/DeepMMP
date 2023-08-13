import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from UNet3D.model import MyNet
from dataset import MyDataset
from miscellaneous import get_local_time_str

abbreviation = {
    'con': 'continuous',
    'discon': 'discontinuous',
    'jus': 'justification',
    'loc': 'location',
    'gen': 'generation',
}


class Manager:
    def __init__(self):
        self.number = get_local_time_str()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bs = 6
        self.fragment_shape = (16, 256, 256)
        self.task_mode = {
            'jus': False,
            'loc': False,
            'gen': True,
        }
        self.n_tasks = self.calc_n_tasks()
        self.task_names = self.gen_task_names()
        self.task_properties = {
            'jus': {
                'input_keys': ['con_t', 'discon_t'],
                'label_keys': ['con_label', 'discon_label'],
                'criterion': 'BCE',
                'loss_coefficient': 1.0,
            },
            'loc': {
                'input_keys': ['discon_t'],
                'label_keys': ['discon_point_label'],
                'criterion': 'CE',
                'loss_coefficient': 1.0,
            },
            'gen': {
                'input_keys': ['even_t'],
                'label_keys': ['odd_t'],
                'criterion': 'MSE',
                'loss_coefficient': 0.01,
            },
            'jus+loc': {
                'input_keys': ['con_t', 'discon_t'],
            },
            'jus+loc+gen': {
                'input_keys': ['con_t', 'discon_t', 'even_t'],
            },
            'loc+gen': {
                'input_keys': ['discon_t', 'even_t'],
            },
        }
        self.optimizer_name = 'Adam'
        # self.writer_path = f'/root/data1/Run/{self.number}'
        # self.weights_path = f'/root/data1/Weight/{self.number}'
        self.writer_path = fr'E:\Code\pythonProjects\sitonHoly\Run\{self.number}'
        self.weights_path = fr'E:\Code\pythonProjects\sitonHoly\Weight\{self.number}'
        self.writer = SummaryWriter(self.writer_path)

        # dataset
        self.dataset_config = {
            # 'images_path': '/root/data1/Dataset1/PIG/images',
            'images_path': r'E:\Code\pythonProjects\sitonHoly\PIG\images',
            'fragment_shape': (16, 256, 256),
            'sub_volume_depth': 8,
        }

        # net
        self.in_channels = 1
        self.out_channels = 1
        self.layer_order = 'bcr'
        self.f_maps = 32
        self.net = self.get_net()

        # optimizer
        self.lr = 0.01
        self.betas = (0.95, 0.99)
        self.eps = 1e-8
        self.momentum = 0.9
        self.optimizer = self.gen_optimizer()

        # training
        self.it = 0
        self.epochs = 50
        self.epoch = 0
        self.val_ratio = 0.2
        self.training = True

        # others
        self.record_per_its = 10

    def gen_loss_func(self, loss_name: str):
        if loss_name == 'CE':
            return nn.CrossEntropyLoss()
        if loss_name == 'MSE':
            return nn.MSELoss()
        if loss_name == 'BCE':
            return nn.BCEWithLogitsLoss()

    def gen_optimizer(self):
        optimizer_name = self.optimizer_name
        if optimizer_name == 'Adam':
            return optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)
        elif optimizer_name == 'SGD':
            return optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)

    def gen_task_names(self) -> str:
        task_names = ''
        for k, v in self.task_mode.items():
            if v:
                if len(task_names):
                    task_names += '+' + k
                else:
                    task_names += k
        return task_names

    def gen_inputs(self, data):
        input_keys = self.task_properties[self.task_names]['input_keys']
        inputs = torch.empty(self.bs, 1, self.fragment_shape[0] // 2, self.fragment_shape[1], self.fragment_shape[2])
        for input_key in input_keys:
            inputs = torch.cat((inputs, data[input_key]), 0)
        return inputs[self.bs:].to(self.device)

    def gen_labels(self, data):
        labels = []
        for k, v in self.task_mode.items():
            if v:
                label_keys = self.task_properties[k]['label_keys']
                label = None
                for idx, label_key in enumerate(label_keys):
                    if not idx:
                        label = data[label_key]
                    else:
                        label = torch.cat((label, data[label_key]), 0)
                labels.append(label.to(self.device))
        return labels

    def calc_losses(self, outputs, labels):
        loss_funcs = []
        loss_coefficients = []
        losses = []
        for k, v in self.task_mode.items():
            if v:
                loss_name = self.task_properties[k]['criterion']
                loss_coefficient = self.task_properties[k]['loss_coefficient']
                loss_funcs.append(self.gen_loss_func(loss_name))
                loss_coefficients.append(loss_coefficient)
        assert len(outputs) == len(loss_funcs) == len(loss_coefficients) == len(labels)
        loss_total = 0.0
        for idx, loss_func in enumerate(loss_funcs):
            loss = loss_func(outputs[idx], labels[idx])
            losses.append(loss)
            loss_total += loss_coefficients[idx] * loss
        return loss_total, losses

    def process_losses(self, loss_total, losses: list):
        self.it += 1
        print(f'iteration {self.it} ending')
        if not (self.it % self.record_per_its == 0):
            return
        self.writer.add_scalar(f"Loss_{'train' if self.training else 'val'}/total", loss_total.item(), self.it)
        for idx, loss in enumerate(losses):
            # self.writer.add_scalars(f'Loss/{str(idx)}', {}, self.it)
            self.writer.add_scalar(f"Loss_{'train' if self.training else 'val'}/{str(idx)}", loss.item(), self.it)

    def get_net(self):
        net = MyNet(self.in_channels, self.out_channels, layer_order=self.layer_order, f_maps=self.f_maps,
                    final_sigmoid=True).to(
            self.device)
        return net

    def get_dataset(self):
        return MyDataset(self.dataset_config)

    def set_train_val(self, status: bool):
        if status:
            self.net.train()
        else:
            self.net.eval()
        self.training = status

    def end_epoch(self):
        self.epoch += 1
        print(f'epoch {str(self.epoch)} ends')

    def calc_n_tasks(self):
        n_tasks = 0
        for k, v in self.task_mode.items():
            if v:
                n_tasks += 1
        return n_tasks

    def end_task(self):
        self.writer.close()


if __name__ == '__main__':
    print(MyDataset(Manager().dataset_config).__getitem__(100)['con_t'].shape)
