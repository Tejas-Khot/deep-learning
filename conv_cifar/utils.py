'''
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: Utility functions for CIFAR dataset
'''

import gzip, cPickle
import os

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np


def load_CIFAR_batch(filename):
    """
    load single batch of CIFAR-10 dataset
    
    @param filename: string of file name in CIFAR
    @return: X, Y: data and labels of images in the CIFAR batch
    """
    with open(filename, 'r') as f:
        datadict=pickle.load(f)
        
        X=datadict['data']
        Y=datadict['labels']
        
        X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y=np.array(Y)
        
        return X, Y


def load_CIFAR10(ROOT):
    """
    load entire CIFAR-10 dataset
    
    @param ROOT: string of data folder
    @return: X_train, Y_train: training data and labels
    @return: X_test, Y_test: testing data and labels
    """
    xs=[]
    ys=[]
    
    for b in range(1,6):
        f=os.path.join(ROOT, "data_batch_%d" % (b, ))
        X, Y=load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
        
    X_train=np.concatenate(xs)
    Y_train=np.concatenate(ys)
    
    del X, Y
    
    X_test, Y_test=load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    
    return X_train, Y_train, X_test, Y_test


def visualize_CIFAR(X_train,
                    Y_train,
                    samples_per_class):
    """
    A visualization function for CIFAR 
    """
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes=len(classes)
    
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(Y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    
    plt.show()