import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pathlib import Path


class JusDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images_path = path
        self.paths = sorted([str(path) for path in Path(self.images_path).glob('*.png')])

    def __len__(self):
        return len(self.paths) // 8

    def __getitem__(self, idx):
        # paths = sorted([str(path) for path in Path(self.images_path).glob(f'{idx}_*')])
        paths = self.paths[idx * 8:(idx + 1) * 8]
        assert len(paths) == 8
        stack = torch.zeros(1, 8, 256, 256)
        for i, path in enumerate(paths):
            one_t = torchvision.io.read_image(path, torchvision.io.ImageReadMode.GRAY)
            stack[0, i] = one_t
        return stack / 255, 1 if idx % 2 == 0 else 0, 'jus'


class LocDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images_path = path
        self.paths = sorted([str(path) for path in Path(self.images_path).glob('*.png')])

    def __len__(self):
        return len(self.paths) // 8 // 2

    def __getitem__(self, idx):
        # paths = sorted([str(path) for path in Path(self.images_path).glob(f'{idx}_*')])
        idx = 2 * idx + 1
        paths = self.paths[idx * 8:(idx + 1) * 8]
        discontinuity_point = (int(Path(paths[0]).stem.split('_')[0]) // 2) % 7
        assert len(paths) == 8
        stack = torch.zeros(1, 8, 256, 256)
        for i, path in enumerate(paths):
            one_t = torchvision.io.read_image(path, torchvision.io.ImageReadMode.GRAY)
            stack[0, i] = one_t
        return stack / 255, discontinuity_point, 'loc'


class GenDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images_path = path
        self.paths = sorted([str(path) for path in Path(self.images_path).glob('*.png')])
        self.transform = transform

    def __len__(self):
        return len(self.paths) // 8 // 2

    def __getitem__(self, idx):
        # paths = sorted([str(path) for path in Path(self.images_path).glob(f'{idx}_*')])
        even_paths = self.paths[idx * 8:(idx + 1) * 8]
        odd_paths = self.paths[2 * idx * 8:(2 * idx + 1) * 8]
        even, odd = torch.zeros(1, 8, 256, 256), torch.zeros(1, 8, 256, 256)
        for i, path in enumerate(even_paths):
            one_even = torchvision.io.read_image(path, torchvision.io.ImageReadMode.GRAY)
            one_odd = torchvision.io.read_image(odd_paths[i], torchvision.io.ImageReadMode.GRAY)
            even[0, i] = one_even
            odd[0, i] = one_odd
        return even / 255, odd / 255, 'gen'


if __name__ == '__main__':
    jus_ds = JusDataset(r'E:\Data\data\Jus\train')
    loc_ds = LocDataset(r'E:\Data\data\Jus\train')
    gen_ds = GenDataset(r'E:\Data\data\Gen\train')

    loc_ds.__getitem__(0)

    concat_ds = ConcatDataset([jus_ds, loc_ds, gen_ds])
    pass
