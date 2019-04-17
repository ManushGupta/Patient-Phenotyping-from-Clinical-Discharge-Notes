import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

import pickle
import ipdb

class CNN(nn.Module):
    def __init__(self, doclen, embsize, pos_wt):
        super(CNN, self).__init__()
        self.doclen = doclen
        self.embsize = embsize
        self.pos_wt = pos_wt

        def conv_pool(in_channel, out_channel, window, width, lenpool):
            conv = nn.Conv2d(1, 1, (window, width))
            pool = nn.AvgPool1d(lenpool)
            return conv, pool

        self.conv1, self.pool1 = conv_pool(1, 1, 2, self.embsize, self.doclen-1)
        self.conv2, self.pool2 = conv_pool(1, 1, 3, self.embsize, self.doclen-2)        
        self.conv3, self.pool3 = conv_pool(1, 1, 4, self.embsize, self.doclen-3)
        self.conv4, self.pool4 = conv_pool(1, 1, 5, self.embsize, self.doclen-4)

        self.aff1 = nn.Linear(4, 1)

    def forward(self, x):
        def conv_block(x, conv, pool):
            x = conv(x).squeeze(-1)
            x = nn.BatchNorm1d(1)(x)
            x = nn.Dropout(p=0.2)(x)
            x = F.relu(x)
            x = pool(x)
            return x.squeeze(-1)

        x1 = conv_block(x, self.conv1, self.pool1)
        x2 = conv_block(x, self.conv2, self.pool2)
        x3 = conv_block(x, self.conv3, self.pool3)
        x4 = conv_block(x, self.conv4, self.pool4)

        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.aff1(x)

        return x

    def loss(self, y_pred, y):
        #  return nn.MultiLabelSoftMarginLoss()(y_pred, y)
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_wt)(y_pred, y)

def evaluate(y_pred, y):
    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)*100
    yhat = y_pred.copy()
    yhat[yhat>=0.07] = 1
    yhat[yhat<0.07] = 0
    err = np.sum(np.abs(yhat - y))
    acc = 100*(1 - err/len(y))
    cmat = confusion_matrix(y, yhat)
    cmat = cmat*100./np.sum(cmat, axis=1)[:,np.newaxis]
    f1 = f1_score(y, yhat)
    return acc, auroc, cmat, f1

if __name__ == '__main__':
    load_model = True
    train_model = False
    test_model = True
    num_epochs = 1
    batch_size = 16

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
        model = CNN(doclen, embsize, pos_wt)
        if load_model:
            model.load_state_dict(torch.load('checkpoints/all/'+conditions[i]+'.pth'))

        # Train the model
        if train_model:
            optm = torch.optim.Adam(model.parameters()) 
            model.train()
            loss_vals = []
            for epoch in range(num_epochs):
                optm.zero_grad()
                # Get data batch
                idx = np.random.choice(np.arange(num_train), size=batch_size)
                batch_x_train = torch.from_numpy(x_train[idx]).float().unsqueeze(1)
                batch_y_train = torch.from_numpy(y_train[idx]).float()
                if epoch == 0:
                    print('Batch dims = ', batch_x_train.shape, batch_y_train.shape)

                # Feed forward
                y_pred = model(batch_x_train)

                # Update weights
                loss = model.loss(y_pred, batch_y_train)
                loss.backward()
                optm.step()
                loss_vals.append(loss.data.numpy())
                #  if epoch%50 == 0:
                print('Loss at epoch {} is {}'.format(epoch, loss.data.numpy()))
                
                del batch_x_train, batch_y_train

            torch.save(model.state_dict(), 'checkpoints/all/'+conditions[i]+'.pth')

        # Test the model
        if test_model:
            model.eval()
            y_pred = model(torch.from_numpy(x_test).float().unsqueeze(1))
            loss = model.loss(y_pred, torch.from_numpy(y_test).float())
            acc, auroc, cmat, f1 = evaluate(y_pred.data.numpy()[:,0], y_test[:,0]) 
            print('Accuracy = {}, AUC = {}, F1 = {}'.format(acc, auroc, f1))
            #  print('Confmatrix: ')
            #  print(cmat)

        # Store results
        #  results[conditions[i]] = {
        #          'pos_wt': pos_wt,
        #          'loss': np.array(loss_vals),
        #          'accuracy': acc,
        #          'auroc': auroc,
        #          'cmat': cmat,
        #          'f1': f1
        #          }
    
    # Save results
    #  with open('outputs/output_all_glove.pickle', 'wb') as f:
    #      pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

   



