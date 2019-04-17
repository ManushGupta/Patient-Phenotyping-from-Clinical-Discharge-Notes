import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

import pickle
import ipdb

class LSTM(nn.Module):
    def __init__(self, doclen, embsize, pos_wt):
        super(LSTM, self).__init__()
        self.doclen = doclen
        self.embsize = embsize
        self.pos_wt = pos_wt
        
        self.lstm = nn.LSTM(embsize, 
                hidden_size=embsize, batch_first=True, 
                bidirectional=False, dropout=0.2)
        self.aff1 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.lstm(x)
        x = x[0][:,-1,:]
        x = self.aff1(x)
        return x

    def loss(self, y_pred, y):
        #  return nn.MultiLabelSoftMarginLoss()(y_pred, y)
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_wt)(y_pred, y)

def evaluate(y_pred, y):
    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)*100
    yhat = y_pred.copy()
    yhat[yhat>=0.1] = 1
    yhat[yhat<0.1] = 0
    err = np.sum(np.abs(yhat - y))
    acc = 100*(1 - err/len(y))
    cmat = confusion_matrix(y, yhat)
    cmat = cmat*100./np.sum(cmat, axis=1)[:,np.newaxis]
    f1 = f1_score(y, y_pred)
    return acc, auroc, cmat, f1

if __name__ == '__main__':
    load_model = False
    train_model = True
    test_model = True
    num_epochs = 2
    batch_size = 32

    # Load data
    data_path = '../../data/data_cnn/'
    x_train = np.load(data_path+'x_train_5k.npz')['data']
    x_test = np.load(data_path+'x_test_5k.npz')['data']
    y_train_all = np.load(data_path+'y_train.npz')['labels']
    y_test_all = np.load(data_path+'y_test.npz')['labels']
    print(x_train.shape, x_test.shape, y_train_all.shape, y_test_all.shape)
    #  ipdb.set_trace()

    conditions = ['Obesity', 'Advanced.Heart.Disease', 'Advanced.Lung.Disease', 'Schizophrenia.and.other.Psychiatric.Disorders', 'Alcohol.Abuse', 'Other.Substance.Abuse', 'Chronic.Pain.Fibromyalgia', 'Chronic.Neurological.Dystrophies', 'Advanced.Cancer', 'Depression']

    results = {}
    for i in range(1):
        
        print('Condition = ', conditions[i])
        rem_labels = lambda x: np.delete(x, [0, 2, 3, 13, 14], axis=1)[:, i:i+1].copy()
        y_train = rem_labels(y_train_all)
        y_test = rem_labels(y_test_all)
        (num_train, doclen, embsize) = x_train.shape

        # Initialize the model
        pos_wt = (len(y_train)-np.sum(y_train))/np.sum(y_train)
        pos_wt = torch.tensor(int(pos_wt)).float()
        print('Pos wt = ', pos_wt)
        model = LSTM(doclen, embsize, pos_wt).cuda()
        #  if load_model:
        #      model.load_state_dict(torch.load('checkpoints/model_heart_weighted.pth'))

        # Train the model
        if train_model:
            optm = torch.optim.Adam(model.parameters()) 
            model.train()
            loss_vals = []
            for epoch in range(num_epochs):
                optm.zero_grad()
                # Get data batch
                idx = np.random.choice(np.arange(num_train), size=batch_size)
                batch_x_train = torch.from_numpy(x_train[idx]).float().cuda()
                batch_y_train = torch.from_numpy(y_train[idx]).float().cuda()
                if epoch == 0:
                    print('Batch dims = ', batch_x_train.shape, batch_y_train.shape)

                # Feed forward
                y_pred = model(batch_x_train)

                # Update weights
                loss = model.loss(y_pred, batch_y_train)
                loss.backward()
                optm.step()
                loss_vals.append(loss.cpu().data.numpy())
                #  if epoch%50 == 0:
                print('Loss at epoch {} is {}'.format(epoch, loss.cpu().data.numpy()))
                
                del batch_x_train, batch_y_train

            torch.save(model.state_dict(), 'checkpoints/lstm/'+conditions[i]+'.pth')

        # Test the model
        if test_model:
            model.eval()
            y_pred = model(torch.from_numpy(x_test).float())
            loss = model.loss(y_pred, torch.from_numpy(y_test).float())
            acc, auroc, cmat, f1 = evaluate(y_pred.data.numpy()[:,0], y_test[:,0]) 
            print('Accuracy = {}, AUC = {}'.format(acc, auroc))
            print('Confmatrix: ')
            print(cmat)

            # Store results
            results[conditions[i]] = {
                    'pos_wt': pos_wt,
                    'loss': np.array(loss_vals),
                    'accuracy': acc,
                    'auroc': auroc,
                    'cmat': cmat,
                    'f1': f1
                    }
    
    # Save results
    with open('outputs/output_all_lstm.pickle', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

