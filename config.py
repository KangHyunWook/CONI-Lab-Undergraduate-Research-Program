import argparse
import torch.optim as optim

optimizer_dict={'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key=='optimizer':
                    value = optimizer_dict[value]

                setattr(self, key, value)

def get_config(**optional_kwargs):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default=r"D:\datasets\Track#3 Imagined speech classification")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--offset', type=int,default=32)
    parser.add_argument('--model', type=str,default='BiLSTM')
    parser.add_argument('--hidden_size', type=int,default=200)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    args = vars(args)
    args.update(optional_kwargs)
    
    return Config(**args)