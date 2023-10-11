import torch

import os
import numpy as np
from datetime import datetime
import argparse

from dataloader.dataloader import data_generator
from model import MTS_LOF

import utils

parser = argparse.ArgumentParser()

SEEDS = [2019, 2020, 2021, 2022, 2023]
#SEEDS = [2021, 2022]

if __name__ == '__main__':
    device = 'cuda:1'
    train_mode = 'finetune'
    data_type = 'sleepEDF'
    data_path = f"./data/{data_type}"

    config_module = __import__(f'config_files.{data_type}_Configs', fromlist=['Config'])
    configs = config_module.Config()
    
    for SEED in SEEDS:
        ckpt = f'./checkpoints/{data_type}/{train_mode}_{SEED}.pth'
    
        os.makedirs(f'./checkpoints/{data_type}/', exist_ok=True)
    
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        train_dl, valid_dl, test_dl = data_generator(data_path, configs, train_mode)
        
        model = MTS_LOF(configs)
        
        if train_mode in ['finetune', 'linear_prob']:
            model.load_state_dict(torch.load(f'./checkpoints/{data_type}/ssl_{SEED}.pth'))
        
        model = model.to(device)
        
        if train_mode == 'linear_prob':
            optimizer = torch.optim.AdamW(model.linear.parameters(), lr=configs.lr, weight_decay=0.05)
        elif train_mode == 'finetune':
            optimizer = torch.optim.AdamW([
                {"params":model.conv_block.parameters(), "lr_mult": 0.1},
                {"params":model.transformer_encoder.parameters(), "lr_mult": 0.1},
                {"params":model.linear.parameters(), "lr_mult": 1.0}],
                lr=configs.lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=0.05)
            
        best_acc = 0.
        
        for epoch in range(1, configs.num_epoch + 1):
            
            print(f'Epoch: {epoch}|{configs.num_epoch} || Seed: {SEED}')
            epoch_loss, epoch_acc = utils.train_epoch(model, train_dl, optimizer, train_mode, device)
            if train_mode in ['supervised', 'linear_prob', 'finetune']:
                val_acc = utils.validate(model, valid_dl, device)
                if best_acc < val_acc:
                    print('Save best acc')
                    best_acc = val_acc
                    torch.save(model.state_dict(), ckpt)
            else:
                torch.save(model.state_dict(), ckpt)
