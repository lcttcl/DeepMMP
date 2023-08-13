import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

mode = 'server'

if mode == 'server':
    root_path = r'/root/data1/Dataset/PIG-230322'
    train_path = r'/root/data1/Dataset/PIG-230322/Jus/train'
    val_path = r'/root/data1/Dataset/PIG-230322/Jus/val'
    gen_train_path = r'/root/data1/Dataset/PIG-230322/Gen/train'
    gen_val_path = r'/root/data1/Dataset/PIG-230322/Gen/val'
    model_save_path = r'/root/data2/New'
else:
    root_path = r'E:\Data\data'
    train_path = r'E:\Data\data\Jus\train'
    val_path = r'E:\Data\data\Jus\val'
    gen_train_path = r'E:\Data\data\Gen\train'
    gen_val_path = r'E:\Data\data\Gen\val'
    model_save_path = r'E:\Data\Weight'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
bs = 16


def train_con(output, label, cur_task, loss_func, i):
    pred = torch.argmax(output, 1)
    correct = (pred == label).sum().float()
    precision = correct / len(label)
    loss = loss_func(output, label)
    print(f'iteration {i}, cur_task:{cur_task}, loss: {loss.item()}, precision: {precision}')
    return loss


def train_sim(output, label, cur_task, loss_func, i):
    loss = loss_func(output, label)
    ssim_total = 0.0
    for opt, lab in zip(output, label):
        vol_a = opt[0].detach().cpu().numpy()
        vol_b = lab[0].detach().cpu().numpy()
        ssim_total += ssim(vol_a, vol_b, data_range=1.0, channel_axis=0)
    print(f'iteration {i}, cur_task:{cur_task}, loss: {loss.item()}, ssim: {ssim_total / len(output)}')
    return loss


def val_con(output, label, loss_func):
    pred = torch.argmax(output, 1)
    correct += (pred == label).sum().float()
    precision = correct / len(label)
    loss = loss_func(output, label)


def val_sim(output, label, loss_func):
    loss = loss_func(output, label)
    return loss


def train_main(dataloader, net, optimizer, epoch):
    net.train()
    for i, data in enumerate(dataloader):
        input = data[0].to(device)
        label = data[1].to(device)
        cur_task = data[2][0]
        loss_func = task_mapping[cur_task]['loss_func']
        train_func = task_mapping[cur_task]['train_func']
        optimizer.zero_grad()
        output = net(input, cur_task)
        loss = train_func(output, label, cur_task, loss_func, i)
        loss.backward()
        optimizer.step()


def val_main(dataloader, loss_func, val_func, net, epoch):
    net.eval()
    print('Validating...')
    with torch.no_grad():
        running_loss = 0.0
        correct = torch.zeros(1).squeeze().to(device)
        for i, data in enumerate(dataloader_val):
            input = data[0].to(device)
            label = data[1].to(device)
            output = net(input)
            loss = val_func(output, label, loss_func)
            running_loss += loss.item()
        loss_epoch = running_loss / len(val_loader)
        precision_epoch = correct / (len(val_loader) * bs)
        print(f'epoch {epoch}, val_avg_loss: {loss_epoch}, val_precision: {precision_epoch}')


task_mapping = {
    'jus': {
        'train_func': train_con,
        'val_func': val_con,
        'loss_func': nn.CrossEntropyLoss(),
    },
    'loc': {
        'train_func': train_con,
        'val_func': val_con,
        'loss_func': nn.CrossEntropyLoss(),
        'classes': list(range(1, 8)),
    },
    'gen': {
        'train_func': train_sim,
        'val_func': val_sim,
        'loss_func': nn.MSELoss(),
    },
}
