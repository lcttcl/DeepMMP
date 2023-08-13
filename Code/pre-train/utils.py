from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import random
import numpy as np

from Dataset import *
from Component import *


def set_seed(seed=None, seed_torch=True):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
      seed : Integer
        A non-negative integer that defines the random state. Default is `None`.
      seed_torch : Boolean
        If `True` sets the random seed for pytorch tensors, so pytorch module
        must be imported. Default is `True`.

    Returns:
      Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def randomize_iter(itr):
    return iter(random.sample(itr, len(itr)))


def save_model(epoch, model, optimizer, task: str):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    filename = f'{model_save_path}/{task}_epoch_{str(epoch).zfill(3)}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


def gen_dataloader(task: str, bs, shuffle):
    print(f'generating dataloader for task {task}')
    ds_train, ds_val = None, None
    if task == 'jus':
        ds_train, ds_val = JusDataset(train_path), JusDataset(val_path)
    elif task == 'loc':
        ds_train, ds_val = LocDataset(train_path), LocDataset(val_path)
    elif task == 'gen':
        ds_train, ds_val = GenDataset(gen_train_path), GenDataset(gen_val_path)
    elif task == 'jus+loc':
        jus_ds_train, jus_ds_val = JusDataset(train_path), JusDataset(val_path)
        loc_ds_train, loc_ds_val = LocDataset(train_path), LocDataset(val_path)
        jus_dl_train, jus_dl_val = DataLoader(jus_ds_train, batch_size=bs, shuffle=shuffle), DataLoader(jus_ds_val,
                                                                                                        batch_size=bs,
                                                                                                        shuffle=shuffle)
        loc_dl_train, loc_dl_val = DataLoader(loc_ds_train, batch_size=bs, shuffle=shuffle), DataLoader(loc_ds_val,
                                                                                                        batch_size=bs,
                                                                                                        shuffle=shuffle)
        merge_dl_train, merge_dl_val = itr_merge(jus_dl_train, loc_dl_train), itr_merge(jus_dl_val, loc_dl_val)
        dl_train, dl_val = randomize_iter(list(merge_dl_train)), randomize_iter(list(merge_dl_val))
        return dl_train, dl_val
    elif task == 'jus+loc+gen':
        jus_ds_train, jus_ds_val = JusDataset(train_path), JusDataset(val_path)
        loc_ds_train, loc_ds_val = LocDataset(train_path), LocDataset(val_path)
        gen_ds_train, gen_ds_val = GenDataset(gen_train_path), GenDataset(gen_val_path)
        jus_dl_train, jus_dl_val = DataLoader(jus_ds_train, batch_size=bs, shuffle=shuffle), DataLoader(jus_ds_val,
                                                                                                        batch_size=bs,
                                                                                                        shuffle=shuffle)
        loc_dl_train, loc_dl_val = DataLoader(loc_ds_train, batch_size=bs, shuffle=shuffle), DataLoader(loc_ds_val,
                                                                                                        batch_size=bs,
                                                                                                        shuffle=shuffle)
        gen_dl_train, gen_dl_val = DataLoader(gen_ds_train, batch_size=bs, shuffle=shuffle), DataLoader(gen_ds_val,
                                                                                                        batch_size=bs,
                                                                                                        shuffle=shuffle)
        merge_dl_train, merge_dl_val = itr_merge(jus_dl_train, loc_dl_train, gen_dl_train), itr_merge(jus_dl_val,
                                                                                                      loc_dl_val,
                                                                                                      gen_dl_val)
        dl_train, dl_val = randomize_iter(list(merge_dl_train)), randomize_iter(list(merge_dl_val))
        return dl_train, dl_val

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=shuffle)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=bs, shuffle=shuffle)
    return dl_train, dl_val
