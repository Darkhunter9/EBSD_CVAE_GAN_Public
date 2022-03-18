"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: data_generator.py

MODULE: util

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Data generator for model training/testing, with choice of image processing
Inherited from Sequence in Keras


Date:
-------------
2022/03/17 ZD 1.0 public version
"""

import numpy as np
import h5py
import cv2
from sklearn.utils import shuffle

import tensorflow as tf

from .eu2qu import eu2qu
from .imgprocess import *


class DataGenerator(tf.keras.utils.Sequence):
    '''
    Data generator for model training/testing, with choice of image processing \\
    Inherited from Sequence in Keras
    '''
    def __init__(self, data, batch_size=32, dim=(60,60), n_channels=1, shuffle=True, processing=None):
        '''
        Initialization of a data generator object

        Input:
        ------
        data: directory to h5 file, string, default: None

        batch_size: number of patterns in a batch sent, int, default: 32

        dim: dimension of patterns, (int, int), default: (60, 60)

        n_channels: number of channels of patterns, int, default: 1

        shuffle: whether to shuffle the order of patterns, bool, default: True

        processing: image processing recipe, string, default: None


        Output:
        ------
        None
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.processing = processing
        self.f = h5py.File(data, 'r')
        self.X = self.f['EMData']['EBSD']['EBSDPatterns'] # patterns from h5 file
        self.y = self.f['EMData']['EBSD']['EulerAngles'] # orientations from h5 file
        print('load data successfully')
        self.on_epoch_end()

        # whether resize is needed
        self.resize = (dim == self.X.shape[1:])

    def __del__(self):
        '''
        Destructor
        '''
        self.close()

    def __len__(self):
        '''
        Denotes the number of batches per epoch

        Input:
        ------
        None


        Output:
        ------
        number of batches per epoch, int
        '''
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        Required by Sequence class

        Input:
        ------
        index: index of batch, int


        Output:
        ------
        X: patterns after processing, 4darray, shape: (n,c,h,w)

        y: labels, 2darray, shape: (n, 4)
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # To make sure indexing elements are in increasing order
        # Otherwise TypeError: Indexing elements must be in increasing order
        indexes = np.sort(indexes)

        # Generate data
        X = np.array(self.X[indexes,:,:])
        X = X.astype(float)

        # resize
        if not self.resize:
            X_new = np.zeros((len(X),self.dim[0],self.dim[1]))
            for i in range(len(X)):
                X_new[i] = cv2.resize(X[i],(self.dim[1],self.dim[0]),interpolation=cv2.INTER_LINEAR)
        else:
            X_new = X
        X_new = X_new.astype(np.uint8)

        # preprocessing
        if self.processing:
            for i in self.processing:
                X_new = eval(i.replace('(','(X_new,',1))

        X_new = X_new.astype(float)
        X_new = np.clip(np.nan_to_num(X_new),0.,255.)
        X_new = X_new / 255.0
        X_new = X_new[:,:,:,np.newaxis]

        y = np.array(self.y[indexes,:])
        y = y.astype(float)
        temp_y = np.zeros((len(y),4))
        for i in range(len(y)):
            temp_y[i] = eu2qu(y[i])
        y = temp_y
        y = np.clip(np.nan_to_num(y),-1.,1.)
        

        # shuffle inside the batch
        if self.shuffle:
            return shuffle(X_new, y)
        else:
            return X_new, y

    def on_epoch_end(self):
        '''
        Update indices after each epoch

        Input:
        ------
        None


        Output:
        ------
        None
        '''
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def close(self):
        '''
        Close h5 file buffer to avoid memory leak

        Input:
        ------
        None


        Output:
        ------
        None
        '''
        self.f.close()


if __name__ == '__main__':
    # Parameters
    params = {'dim': (480,640),
            'batch_size': 32,
            'n_channels': 1,
            'shuffle': False}

    # Generators
    imgprocesser = imgprocess(recipe=['equalization(8, (10, 10))','circularmask()'])
    training_generator = DataGenerator(data='Ni_testing.h5', batch_size=1, dim=(60,60), n_channels=1,
                                        shuffle=False, processing=imgprocesser)
    
    X, y = training_generator[0]
    X = X[0]*255.
    X.resize((60,60))
    X.astype('uint8')
