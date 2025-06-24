import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        n_channels=64
        fc_size=1024
        output_size=5

        self.input_size=input_size=config.offset
        self.hidden_size=hidden_sizes=[config.hidden_size,fc_size,output_size]
        rnn = nn.LSTM
        self.eeg_rnn1= rnn(input_size, hidden_sizes[0], bidirectional=True)
        self.eeg_rnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        self.layer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        
        self.project_eeg=nn.Sequential()
        self.project_eeg.add_module('project_eeg', nn.Linear(in_features=4*hidden_sizes[0]*n_channels,out_features=hidden_sizes[1]))
        self.project_eeg.add_module('project_eeg_activation', nn.ReLU())
        self.project_eeg.add_module('project_eeg_layer_norm', nn.LayerNorm(hidden_sizes[1]))
        self.project_eeg.add_module('project_eeg_2',nn.Linear(in_features=hidden_sizes[1],out_features=hidden_sizes[2]))

        

    def extract_features(self, x, batch_size):
        
        x=torch.unsqueeze(x,dim=0)
        
        
        packed_h1, (h1, _) = self.eeg_rnn1(x)
        normed_h1 = self.layer_norm(packed_h1)
        
        _, (h2, _)=self.eeg_rnn2(normed_h1)


        o=torch.cat((h1, h2), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)    
        
        return o
    
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(x.shape[1]):
            ch_x=x[:,i,:]
            o = self.extract_features(ch_x, batch_size)
            outputs.append(o)

        final_output = torch.stack(outputs, dim=1)
        flattend = final_output.view(batch_size,-1)
        final_out = self.project_eeg(flattend)


        return final_out