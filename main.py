from solver import Solver
from config import get_config
from create_dataset import BCIComp
from data_loader import get_loader

import torch.backends.cudnn as cudnn
import random
import torch
import os

if __name__=='__main__':
    cuda = True
    cudnn.benchmark = True
    manual_seed = 123
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
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
    test_loss, test_acc = solver.eval(mode="test", to_print=True)
    with open(train_config.save_path, train_config.w_mode) as f:
        if train_config.w_mode=='w':
            f.write('optimizer,learning rate,hidden_size,accuracy\n')
        f.write(str(train_config.optimizer)+','+str(train_config.learning_rate)+','+ str(train_config.hidden_size)+','+str(test_acc)+'\n')
    f.close()
    print('test acc:', test_acc)
    
    
    