import pickle
import pandas as pd
import scipy.io as sio
import numpy as np
import os
import mat73
import random

def to_pickle(obj,path):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class BCIComp:
    def __init__(self, config):
        DATA_PATH=str(config.dataset_dir)
        offset=config.offset

        try:
            self.train = load_pickle('./train.pkl')
            self.val = load_pickle('./val.pkl')
            self.test = load_pickle('./test.pkl')
        except:
            train_data_folder=os.path.join(DATA_PATH,'Training set')   
            train_items = os.listdir(train_data_folder)                                                  
            train=[]    
            for item in train_items:
                full_path=os.path.join(train_data_folder, item)
                total_data = sio.loadmat(full_path)
                features = total_data['epo_train']['x'][0][0]
                train_labels = np.argmax(total_data['epo_train']['y'][0][0]==1,axis=0)
                
                features = features.transpose(2,1,0)
                dimensionality = features.shape[2]
                
                for i in range(features.shape[0]):
                    trial_feature=features[i]
                    trial_label=train_labels[i]
                    for j in range(0, offset*(dimensionality//offset),offset):
                        seg=trial_feature[:,j:j+offset]
                        train.append((seg, trial_label))
                        
            random.shuffle(train)
            train_len=int(0.6*len(train))

            test=train[train_len:]
            train=train[:train_len]
            
            val_len = int(0.3*len(test))
            val=test[val_len:]
            test=test[:val_len]

            to_pickle(train, 'train.pkl')
            to_pickle(val, 'val.pkl')
            to_pickle(test, 'test.pkl')

            self.train = train
            self.val = val
            self.test = test
            

    def get_data(self, mode):
        if mode=='train':
            return self.train
        elif mode=='val':
            return self.val
        elif mode=='test':
            return self.test
        else:
            print('Error mode')
            exit(1)