from create_dataset import BCIComp
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
    
    
    