import pickle
import pandas as pd
import scipy.io as sio
import numpy as np
import os
import mat73

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
            train=[]    
            val=[]
            test=[]
            
            train_data_folder=os.path.join(DATA_PATH,'Training set')
            val_data_folder = os.path.join(DATA_PATH, 'Validation set')
            test_data_folder = os.path.join(DATA_PATH, 'Test set')
                         
            train_items = os.listdir(train_data_folder)
            val_items = os.listdir(val_data_folder)
            test_items = os.listdir(test_data_folder)[:15]
                         
                         
            #set train data
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
                        

            #set validation data
            for item in val_items:
                full_path=os.path.join(val_data_folder, item)
                total_data = sio.loadmat(full_path)
                
                features = total_data['epo_validation']['x'][0][0]
                val_labels = np.argmax(total_data['epo_validation']['y'][0][0]==1,axis=0)
                
                features = features.transpose(2,1,0)
                dimensionality = features.shape[2]
                
                for i in range(features.shape[0]):
                    trial_feature=features[i]
                    trial_label=val_labels[i]
                    for j in range(0, offset*(dimensionality//offset),offset):
                        seg=trial_feature[:,j:j+offset]
                        val.append((seg, trial_label))



            #set test data
            test_groundtruth_fpath=os.path.join(test_data_folder,"Track3_Answer Sheet_Test.xlsx")
            df = pd.read_excel(test_groundtruth_fpath)
           
            for item in test_items:
                test_data = mat73.loadmat(os.path.join(test_data_folder, item))
                test_data = test_data['epo_test']['x']
                
                test_data = test_data.transpose(2,1,0)
                
                dimensionality = test_data.shape[0]
                splits = item.split('.')
                index = splits[-2].index('ple')
                current_test_label_idx = int(splits[-2][index+3:])*2
                
                test_labels = df.iloc[:, current_test_label_idx]
                test_labels = test_labels.to_numpy()[2:]-1
                
                for i in range(test_data.shape[0]):
                    trial_data=test_data[i]
                    trial_label=test_labels[i]
                    for j in range(0,offset*(dimensionality//offset),offset):
                        seg = trial_data[:,j:j+32]
                        
                        test.append((seg,trial_label))
            
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