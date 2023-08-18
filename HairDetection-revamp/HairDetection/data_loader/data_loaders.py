import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from HairDetection.base import DataLoaderBase




# Update and use our own custom dataset
class CustomDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using DataLoaderBase
    """
    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers,
                 train=True):
        self.data_dir = data_dir

        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True)
        )
        self.valid_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        ) if train else None

        self.init_kwargs1 = {
            'batch_size': batch_size
        }

        self.init_kwargs = {
            'batch_size': batch_size,
            # 'data_dir': data_dir,
            # 'shuffle': shuffle,
            # 'validation_split': validation_split,
            # 'nworkers': nworkers,
            # 'data_dir': 'data/MNIST/raw/'
        }
        super().__init__(self.train_dataset, shuffle=shuffle, num_workers=nworkers, **self.init_kwargs1)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset,shuffle=False, **self.init_kwargs)
