from lime import lime_tabular
from config import get_config
from torch.utils.data import Dataset, DataLoader
from create_dataset import BCIComp

import models
import mne
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def get_feature_weight_map(local_explanation):
    feature_count_map={}
    feature_weight_map={}

    for feature, weight in local_explanation:
        
        #weight=abs(weight)
        splits=feature.split('<')
        for i in range(len(splits)):
            if splits[i].find('t-')>0:
                channel=splits[i].split('_')[0].strip()
                
                if channel not in feature_weight_map:
                    feature_weight_map[channel]=0.0
                    feature_count_map[channel]=0

                feature_weight_map[channel]+=weight
                feature_count_map[channel]+=1

    for key in feature_weight_map:
        feature_weight_map[key]=feature_weight_map[key]/feature_count_map[key]
    
            
    return feature_weight_map        

def get_index_weight_map(feature_weight_map):

    index_weight_map={}

    for ch in ch_names:
        if ch in feature_weight_map:
            ch_idx= ch_names.index(ch)
            index_weight_map[ch_idx]=feature_weight_map[ch]
    
    return index_weight_map

def get_weighted_test_sample(sample, index_weight_map):
    
    for i in range(len(sample[:,0])):
        if i in index_weight_map:
            # sample[i,0]*=index_weight_map[i]
            sample[i,0]=0
            sample[i,0]=index_weight_map[i]
    
    return sample[:,0] #delta

def predict_proba(X):
    inputs=torch.from_numpy(X).float()

    # inputs = torch.unsqueeze(inputs, axis=1)
    

    inputs=inputs.to(device)

    outputs = model(inputs)
   
    probabilities = torch.softmax(outputs, dim=1)

    return probabilities.detach().cpu().numpy()

def get_explanation(X, y, sample):

    explainer = lime_tabular.RecurrentTabularExplainer(X, training_labels=y, feature_names=channel_orders,
                                                       discretize_continuous=True,
                                                       class_names=[0,1,2,3,4],                       
                                                       discretizer='decile')

    explanation = explainer.explain_instance(sample, predict_proba, num_features=100, num_samples=310)

    return explanation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_config = get_config(mode='test')

model = getattr(models, 'BiLSTM')(test_config).to(device)
model.load_state_dict(torch.load('./checkpoints/model_best.std'))


#todo: load test data to plot LIME detected activations
channel_orders=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 
'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz','O2','PO10','AF7','AF3',
'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5',
'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
'''
['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 
'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 
'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1'
, 'P2', 'P6', 'P07', 'PO3', 'POz', 'PO4', 'PO8']
'''
test_dataloader=get_loader(test_config, shuffle=False)
montage_1020 = mne.channels.make_standard_montage('standard_1020')

ch_names = [ch for ch in montage_1020.ch_names if ch in channel_orders]


# for i in range(len(channel_orders)):
    # print(i+1, channel_orders[i])
# print('len:', len(channel_orders))
# exit()
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(12,3))
info = mne.create_info(ch_names=channel_orders, sfreq=256., ch_types='eeg')

hello_list=[] # 0
help_list=[] # 1
stop_list=[] # 2
thank_list=[] # 3
yes_list=[] # 4

all_data=[]
all_labels=[]

for X_train, y_train in test_dataloader:
    #todo: group 0, 1, 2,3, 4
    X_train=X_train.cpu().numpy()
    
    y_train=y_train.cpu().numpy()
    for i in range(y_train.shape[0]):
        if y_train[i]==0:
            hello_list.append(X_train[i])
        elif y_train[i]==1:
            help_list.append(X_train[i])
        elif y_train[i]==2:
            stop_list.append(X_train[i])
        elif y_train[i]==3:
            thank_list.append(X_train[i])
        elif y_train[i]==4:
            yes_list.append(X_train[i])
        all_data.append(X_train[i])
        all_labels.append(y_train[i])

hello_list=np.asarray(hello_list)
help_list=np.asarray(help_list)
stop_list=np.asarray(stop_list)
thank_list=np.asarray(thank_list)
yes_list=np.asarray(yes_list)

all_data=np.asarray(all_data)
all_labels=np.asarray(all_labels)
all_data_mean = np.mean(all_data, axis=2)
all_data_mean=all_data_mean.transpose(1,0)

evoked = mne.EvokedArray(all_data_mean, info)
evoked.set_montage(montage_1020)
import random

def help_compute_mean(data,indices):
    ten_weights_list=[]
    random_list=np.arange(len(indices))

    random.shuffle(random_list)

    for i in range(10):
        idx=indices[random_list[i]]
        
        explanation = get_explanation(all_data, all_labels, data[idx])
        local_explanation = explanation.as_list()
        feature_weight_map = get_feature_weight_map(local_explanation)
        
        index_weight_map = get_index_weight_map(feature_weight_map)
        weighted_sample = get_weighted_test_sample(data[idx], index_weight_map)
    
        ten_weights_list.append(weighted_sample)

    return ten_weights_list

#return 10 random whose predictions is correct
def compute_mean(data, ground_truth):

    preds = model(torch.from_numpy(data).to(device).float())
    preds=preds.detach().cpu().numpy()
    preds=np.argmax(preds,axis=1)
    
    correct_indices=np.where((preds==ground_truth))[0]
    incorrect_indices=np.where((preds!=ground_truth))[0]
    
    ten_correct_weights_list=help_compute_mean(data,correct_indices)
    ten_incorrect_weights_list=help_compute_mean(data,incorrect_indices)
    

    return ten_correct_weights_list, ten_incorrect_weights_list

hello_correct_weights, hello_incorrect_weights = compute_mean(yes_list,4)


for i in range(len(hello_correct_weights)):
    im1, cn1 = mne.viz.plot_topomap(hello_correct_weights[i], evoked.info, show=False, axes=axes[0,i], res=1200)
    im1, cn1 = mne.viz.plot_topomap(hello_incorrect_weights[i], evoked.info, show=False, axes=axes[1,i], res=1200)
    axes[0,i].set_title('Sample '+str(i+1))

axes[0,0].set_ylabel('Correct')
axes[1,0].set_ylabel('Incorrect', color='red')
fig.subplots_adjust(bottom=0.5)
fig.text(0.04, -0.09, '(a) Comparison of LIME detected brain regions of correctly and incorrectly predicted samples when imagining yes',
         ha='left', va='bottom', fontsize=14)

plt.tight_layout()
plt.savefig('lime.png',bbox_inches='tight')
plt.show()
print('ch_names:', ch_names)
# plt.show()
exit()