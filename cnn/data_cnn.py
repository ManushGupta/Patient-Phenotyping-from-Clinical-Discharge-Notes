import numpy as np
import ipdb
import h5py
import pandas as pd
from tqdm import tqdm
from time import time

def load_bin_vec(filename):
    """
    Loads a word2vec file and creates word2idx
    :param filename: The name of the file with word2vec vectors
    :return: word2vec dictionary, the size of embeddings and number of words in word2vec
    """
    w2v = {}
    with open(filename, 'r') as f:
        header = f.readline()
        vocab_size, emb_size = map(int, header.split())
        for line in f:
            cline = line.split()
            w2v[cline[0]] = np.array(cline[1:], dtype=np.float64)
    return w2v, emb_size, vocab_size

def get_word(idx, word_dict):
    return word_dict[word_dict['idx']==idx]['word'].iloc[0]

def get_data(data_path=None):
    if data_path==None:
        path_w2v = '../../data/glove.txt'
        path_data = '../data.h5'
        path_dict = '../words.dict'

        # get word2vec features for 470260 words in complete MIMIC notes data
        word2vec, emb_size, v_large = load_bin_vec(path_w2v)
        
        # Load the data : train_x = (1127, 10209)
        with h5py.File(path_data, "r") as f:
            train_x = f["train"][:, :5000]
            test_x = f["test"][:, :5000]
            #  train_y = f["train_label"][:]
            #  test_y = f["test_label"][:]
        #  ipdb.set_trace()
        word_dict = pd.read_csv(path_dict, sep=' ')
        word_dict.columns = ['word', 'idx']
        words = word_dict['word'].values
        
        # Get the w2v data : train_x = (1127, 10209, 50)
        #  train_x = test_x
        (num_train, doclen) = train_x.shape
        train_x = train_x - 2
        train_w2v = np.zeros((num_train, doclen, emb_size))
        num_words = len(words)
        #  np.savez_compressed('../data/data_cnn/y_train.npz', labels=train_y)
        #  np.savez_compressed('../data/data_cnn/y_test.npz', labels=test_y)
        for i in tqdm(range(num_words)):
            if words[i] in word2vec:
                train_w2v[np.where(train_x==i)] = word2vec[words[i]]
        np.savez_compressed('../../data/data_cnn/x_train_glove.npz', data=train_w2v)

        (num_test, doclen) = test_x.shape
        test_x = test_x - 2
        test_w2v = np.zeros((num_test, doclen, emb_size))
        num_words = len(words)
        #  np.savez_compressed('../data/data_cnn/y_test.npz', labels=test_y)
        #  np.savez_compressed('../data/data_cnn/y_test.npz', labels=test_y)
        for i in tqdm(range(num_words)):
            if words[i] in word2vec:
                test_w2v[np.where(test_x==i)] = word2vec[words[i]]
        np.savez_compressed('../../data/data_cnn/x_test_glove.npz', data=test_w2v)


        return train_w2v

    else:
        return np.load(data_path)['data']
    
if __name__=='__main__':
    t = time()
    data = get_data()#'../data/data_cnn_5k.npz')
    #  labels =
    print(data.shape, time()-t)





