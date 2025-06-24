from solver import Solver
from config import get_config
from create_dataset import BCIComp

import torch
import os


from torch.utils.data import Dataset, DataLoader

class ISDataset(Dataset):
    def __init__(self,config):
        dataset = BCIComp(config)
        
        self.data=dataset.get_data(config.mode)
        self.len=len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len

def get_loader(config, shuffle=True):
    dataset = ISDataset(config)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle
    )
    
    return dataloader

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