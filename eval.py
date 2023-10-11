import torch

import os
import numpy as np
from datetime import datetime
import argparse

from dataloader.dataloader import data_generator
from model import MTS_LOF

import utils

from sklearn.metrics import classification_report, accuracy_score

SEEDS = [2019, 2020, 2021, 2022, 2023]
#SEEDS = [2019]

if __name__ == '__main__':
    device = 'cuda:1'
    data_type = 'sleepEDF'
    data_path = f"./data/{data_type}"

    train_mode = 'finetune'
    
    config_module = __import__(f'config_files.{data_type}_Configs', fromlist=['Config'])
    configs = config_module.Config()
    
    F1 = []
    ACC = []
    for SEED in SEEDS:
        train_dl, valid_dl, test_dl = data_generator(data_path, configs, 'supervised')
        
        ckpt = f'./checkpoints/{data_type}/{train_mode}_{SEED}.pth'
        #ckpt = f'./checkpoints/semi/{data_type}/{train_mode}_50_{SEED}.pth'
        model = MTS_LOF(configs)
        model.load_state_dict(torch.load(ckpt))
        model = model.to(device)
        model.eval()
        
        pred_labels = np.array([])
        true_labels = np.array([])
        
        print('*****************test*******************')
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_dl, start=1):
                x = x.float().to(device)
                y = y.to(device)
                
                outputs, _ = model(x)
                
                pred = outputs.detach().cpu().argmax(dim=1)
                true_targets = y.cpu()
                
                pred_labels = np.append(pred_labels, pred.numpy())
                true_labels = np.append(true_labels, y.data.cpu().numpy())
                
        pred_labels = np.array(pred_labels).astype(int)
        true_labels = np.array(true_labels).astype(int)
        
        r = classification_report(true_labels, pred_labels, output_dict=True)
        accuracy = accuracy_score(true_labels, pred_labels)
        
        ACC.append(accuracy* 100)
        F1.append(r["macro avg"]["f1-score"] * 100)
    
    
    print(f'Accuracy: {np.mean(ACC)} $\pm$ {np.std(ACC)}')
    print(f'F1: {np.mean(F1)} $\pm$ {np.std(F1)}')