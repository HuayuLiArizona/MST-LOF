import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
import numpy as np
from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in the second dimension
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float()  # Convert to PyTorch tensor
            self.y_data = torch.from_numpy(y_train).long()   # Convert to PyTorch tensor
        else:
            self.x_data = X_train
            self.y_data = y_train
        
        self.len = X_train.shape[0]
        
        self.jitter_ratio = config.augmentation.jitter_ratio
        self.jitter_scale_ratio = config.augmentation.jitter_scale_ratio
        self.max_segments = config.augmentation.max_seg
    
    def __getitem__(self, index):
        x, y = self.x_data[index], self.y_data[index]
        
        if self.training_mode == "ssl":
            x = x + self.jitter_ratio*torch.randn(x.size())
            
        return x, y

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, 'supervised')
    test_dataset = Load_Dataset(test_dataset, configs, 'supervised')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader

def data_generator_semi(data_path, configs, percentage):
    
    
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    
    num_samples = len(train_dataset['samples'])
    keep_size = int((percentage*0.01) * num_samples)
    drop_size = num_samples - keep_size
    
    print(f"Total samples: {num_samples} || Keep samples {keep_size}")
    
    train_dataset = Load_Dataset(train_dataset, configs, 'supervised')
    train_dataset, _ = random_split(train_dataset, [keep_size, drop_size])
    
    valid_dataset = Load_Dataset(valid_dataset, configs, 'supervised')
    test_dataset = Load_Dataset(test_dataset, configs, 'supervised')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
