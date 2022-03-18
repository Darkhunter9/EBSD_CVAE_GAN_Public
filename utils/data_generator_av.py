"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: data_generator_av.py

MODULE: util

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Data generator including multiple attributes (orientation + av)
for model training/testing, with choice of image processing
Inherited from DataGenerator


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
from .data_generator import DataGenerator

# maximum accelerating voltage: 30kV
AV_MAX = 30.

class DataGeneratorAV(DataGenerator):
    '''
    Data generator for model training/testing, including multiple attributes (orientation + av)
    with choice of image processing \\
    Inherited from DataGenerator class
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
        super().__init__(data, batch_size, dim, n_channels, shuffle, processing)
        self.av = self.f['EMData']['EBSD']['AcceleratingVoltage']


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

        y: labels, 2darray, shape: (n, 5) (orientation + av)
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
            # self.processing = ('clahe(10, (10, 10))','squaremask()')
            for i in self.processing:
                X_new = eval(i.replace('(','(X_new,',1))

        X_new = X_new.astype(float)
        X_new = np.clip(np.nan_to_num(X_new),0.,255.)
        X_new = X_new / 255.0
        X_new = X_new[:,:,:,np.newaxis]

        # orientation
        y = np.array(self.y[indexes,:])
        y = y.astype(float)
        temp_y = np.zeros((len(y),4))
        for i in range(len(y)):
            temp_y[i] = eu2qu(y[i])
        y = temp_y
        y = np.clip(np.nan_to_num(y),-1.,1.)
        

        # av
        av = np.array(self.av[indexes])
        av = av.astype(float) / AV_MAX
        av = av[:,np.newaxis]
        y = np.concatenate([y, av], axis=1)

        # shuffle inside the batch
        if self.shuffle:
            return shuffle(X_new, y)
        else:
            return X_new, y