from sklearn.metrics import accuracy_score

import numpy as np
import os
import torch
import models
import torch.nn as nn

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_dataloader,val_dataloader, test_dataloader, is_train=True, model=None):
        self.train_config=train_config
        self.epoch_i = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_train = is_train
        self.model = model
    
    def build(self, cuda=True):
        
        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()
            
            
    def train(self):
        best_val_loss=float('inf')
        curr_patience = patience = self.train_config.patience
        self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        for e in range(self.train_config.n_epoch):
            self.model.train()
            for features, labels in self.train_dataloader:
                features=features.float()
                features=features.cuda(0)
                labels = labels.cuda(0)
                self.model.zero_grad()
                outputs = self.model(features)
                
                loss = criterion(outputs, labels)

                loss.backward()
                
                self.optimizer.step()

            val_loss, val_acc = self.eval(mode="val")
            print('epoch: {0} | val loss: {1} | val acc {2}'.format(e+1, round(val_loss,3), round(val_acc*100,2)))

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_best.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_best.std')
                curr_patience = patience
            else:
                curr_patience-=1
                if curr_patience<=-1:
                    print("Running out of patience, loading previous best model.")
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_best.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_best.std'))
                    
                    print("Running out of patience, early stopping.")
                    break
                
        test_loss, test_acc = self.eval(mode="test", to_print=True)
        print('test acc:', test_acc)

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):

        test_preds = np.argmax(y_pred, 1)
        test_truth = y_true

        if to_print:
            print("Accuracy:", accuracy_score(test_truth, test_preds))
        
        return accuracy_score(test_truth, test_preds)


    
    def eval(self, mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()
        
        y_true, y_pred = [], []
        eval_loss=[]
        
        if mode == "val":
            dataloader = self.val_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_best.std'))
            

        with torch.no_grad():

            for features, labels in dataloader:
                self.model.zero_grad()
                features = features.float()
                features = features.cuda(0)
                labels = labels.cuda(0)
                
                outputs = self.model(features)

                labels = labels.squeeze()
                
                loss = self.criterion(outputs, labels)
                
                eval_loss.append(loss.item())
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy