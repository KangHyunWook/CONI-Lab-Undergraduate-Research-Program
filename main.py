from solver import Solver
from config import get_config
from create_dataset import BCIComp
from data_loader import get_loader

import torch
import os

if __name__=='__main__':
    #create_loader
    train_config = get_config(mode='train')
    val_config = get_config(mode='val')
    test_config = get_config(mode='test')
    device="cuda:0"
    train_dataloader=get_loader(train_config, shuffle=True)
    val_dataloader = get_loader(val_config, shuffle=True)
    test_dataloader=get_loader(test_config, shuffle=False)
    
    #todo make solver and call train
    solver = Solver
    solver = solver(train_config, val_config, test_config, train_dataloader, val_dataloader, test_dataloader, is_train=True)
    
    solver.build()
    
    solver.train()