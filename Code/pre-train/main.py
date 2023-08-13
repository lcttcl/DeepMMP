from Dataset import *
from Component import *
from Net.model import MyNet
import torch.optim as optim
import torch.nn as nn
from utils import save_model, gen_dataloader, set_seed
import itertools

set_seed(2023, seed_torch=True)

net = MyNet(1, 1, layer_order='gcr', f_maps=32, final_sigmoid=True, n_cls=7).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# task = 'jus'
# task = 'loc'
# task = 'jus+loc'
# task = 'gen'
task = 'jus+loc+gen'
for epoch in range(50):
    dataloader_train, dataloader_val = gen_dataloader(task, bs, shuffle=True)
    print(f'task: {task}, starting epoch {epoch + 1}')
    train_main(dataloader_train, net, optimizer, epoch)
    # val_main(dataloader_val, val_func, loss_func, net, epoch)

save_model(50, net, optimizer, task)
