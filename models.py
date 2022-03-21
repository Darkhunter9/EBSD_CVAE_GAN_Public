"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: models.py

MODULE: CVAE_GAN

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Build EBSD_GAN models.

Date:
-------------
2022/03/20 ZD 1.0 public version
"""

import os
import sys
import time
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.random import normal
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Conv2DTranspose, Activation, Lambda, Dense, Dropout, Flatten, ReLU, LeakyReLU, Reshape, Concatenate, Add
from tensorflow.keras.constraints import MaxNorm
from tensorflow.debugging import assert_equal



class EBSD_GAN:
    '''
    EBSD_CVAE_GAN model class \\
    Can be easily turned into a pure CVAE/GAN model
    '''
    def __init__(self, img_size=(60,60,1), batch_size=64, latent_size=100, att_size=4):
        '''
        Initialization of a data generator object

        Input:
        ------
        img_size: shape of images, (h, w, c), (int, int, int), default: (60, 60, 1)

        batch_size: number of patterns in a batch sent, int, default: 64

        latent_size: dimension of latent representation, int, default: 100

        att_size: dimension of attribute tensor, int, default: 4 (quaternion)


        Output:
        ------
        None
        '''
        self.img_size = img_size
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.att_size = att_size

        self.build_Generator()
        self.build_Discriminator()

        # print out model architecture, for debug
        # self.Enc.summary()
        # self.Dec.summary()
        # self.G.summary()
        # self.D.summary()


    @tf.function
    def sample(self, orientations, eps=None, mean=None, logvar=None, training=True):
        '''
        Sample a batch of patterns \\
        If eps is None, latent representation will be sampled from unit normal distribution \\
        Otherwise mean and logvar should not be None, latent representation will be mean + exp(.5 * logvar) * eps

        Input:
        ------
        orientations: orientation or the whole attribute tensor, 2dtensor, shape: (n, att_size)

        eps: latent representation, None or 2dtensor, shape: (n, latent_size)

        mean: mean value of latent normal distribution, None or 1dtensor, shape: (latent_size,)

        logvar: log of variance of latent normal distribution, None or 1dtensor, shape: (latent_size,)

        training: training mode, bool, default: True


        Output:
        ------
        pattern: generated EBSD pattern, 4dtensor, shape: (n,) + img_size
        '''
        batch_size = tf.shape(orientations)[0]

        orientations = tf.cast(orientations, dtype=tf.float32)
        
        if eps is None:
            eps = normal(shape=(batch_size, self.latent_size), dtype=tf.float32)
        
        if mean is not None and logvar is not None:
            eps = tf.add(tf.multiply(eps, tf.exp(tf.multiply(logvar, .5))), mean)

        return self.Dec([eps, orientations], training=training)


    def build_Enc(self):
        '''
        Build encoder part in CVAE

        Input:
        ------
        None


        Output:
        ------
        None
        '''
        x = inputs_x = Input(self.img_size, name='Enc_input', dtype=tf.float32)
        inputs_o = Input(self.att_size, name='Enc_orientation_input', dtype=tf.float32)
        # x = tf.random.shuffle(x)

        # Encoder
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block1_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block1_conv2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)

        residual2 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Enc_residual2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block2_conv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block2_conv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='Enc_add2')([x, residual2])
        x = LeakyReLU(alpha=0.2)(x)

        residual3 = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Enc_residual3', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block3_conv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block3_conv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='Enc_add3')([x, residual3])
        x = LeakyReLU(alpha=0.2)(x)

        residual4 = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='Enc_residual4', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = SeparableConv2D(728, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block4_conv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = SeparableConv2D(728, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='Enc_block4_conv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='Enc_add4')([x, residual4])
        x = LeakyReLU(alpha=0.2)(x)

        residual5 = Conv2D(512, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='Enc_residual5', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Enc_block5_conv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = SeparableConv2D(512, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='Enc_block5_conv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='Enc_add5')([x, residual5])
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten(name='Enc_flatten')(x)
        x = Concatenate(axis=-1, name='Enc_concat')([x, inputs_o])
        x = Dense(2048, name='Enc_dense1', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(2048, name='Enc_dense2', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1024, name='Enc_dense3', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = Dense(self.latent_size*2, name='Enc_dense4')(x)

        mean, logvar = tf.split(x, num_or_size_splits=2, axis=-1)

        self.Enc = models.Model([inputs_x, inputs_o], [mean, logvar], name='Encoder')


    def build_Dec(self):
        '''
        Build decoder part in CVAE \\
        The subnet also serves as generator in GAN

        Input:
        ------
        None


        Output:
        ------
        None
        '''
        y = inputs_y = Input(self.latent_size, name='Dec_input', dtype=tf.float32)
        inputs_o = Input(self.att_size, name='Dec_orientation_input', dtype=tf.float32)

        y = Dense(1024, name='Dec_dense1', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Dense(2048, name='Dec_dense2', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Dense(4096, name='Dec_dense3', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Dense((self.img_size[0]//4)*(self.img_size[1]//4)*256, name='Dec_dense4', kernel_constraint=MaxNorm(5), bias_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)

        o = Dense(1024, name='Dec_dense5')(inputs_o)
        o = Dense(2048, name='Dec_dense6')(o)
        o = LeakyReLU(alpha=0.2)(o)
        o = Dense(4096, name='Dec_dense7')(o)
        o = LeakyReLU(alpha=0.2)(o)
        o = Dense((self.img_size[0]//4)*(self.img_size[1]//4)*256, name='Dec_dense8')(o)
        o = LeakyReLU(alpha=0.2)(o)

        y = Add(name='Dec_concat')([y, o])

        y = Reshape(target_shape=(self.img_size[0]//4, self.img_size[1]//4, 256))(y)

        y = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block1_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block1_conv2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)

        residual2 = Conv2DTranspose(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Dec_residual2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block2_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block2_conv2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Add(name='Dec_add2')([y, residual2])
        y = LeakyReLU(alpha=0.2)(y)

        residual3 = Conv2DTranspose(728, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='Dec_residual3', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Conv2DTranspose(728, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block3_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2DTranspose(728, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='Dec_block3_conv2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Add(name='Dec_add3')([y, residual3])
        y = LeakyReLU(alpha=0.2)(y)

        residual4 = Conv2DTranspose(512, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='Dec_residual4', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block4_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='Dec_block4_conv2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Add(name='Dec_add4')([y, residual4])

        y = Conv2DTranspose(self.img_size[-1], (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Dec_block5_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Activation('sigmoid')(y)

        self.Dec = models.Model([inputs_y, inputs_o], y, name='Decoder')


    def build_Generator(self):
        '''
        Build the whole CVAE part

        Input:
        ------
        None


        Output:
        ------
        None
        '''
        self.build_Enc()
        self.build_Dec()

        inputs_x = Input(self.img_size, name='G_input', dtype=tf.float32)
        inputs_o = Input(self.att_size, name='G_orientation_input', dtype=tf.float32)

        # Note: should avoid using tfp.layers.IndependentNormal in G
        # input of tfp.layers.IndependentNormal is of shape (batch_size, mean+var)
        # according to source code, var will be processed using softplus to avoid negative value
        
        # whole Generator
        # concat orientation with sampled z
        mean, logvar = self.Enc([inputs_x, inputs_o])
        mean_logvar = Concatenate(axis=-1)([mean, tf.math.exp(logvar*.5)])
        # eps = tfp.layers.IndependentNormal(self.latent_size)(mean_logvar)
        eps = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Independent(tfp.distributions.Normal(loc=t[:,:self.latent_size], scale=t[:,self.latent_size:])))(mean_logvar)

        # decode
        y = self.Dec([eps, inputs_o])
        self.G = models.Model([inputs_x, inputs_o], [mean, logvar, y], name='G')


    def build_Discriminator(self):
        '''
        Build discriminator part in GAN

        Input:
        ------
        None


        Output:
        ------
        None
        '''
        # Input
        x = inputs_x = Input(self.img_size, name='D_input')

        # start flow
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='D_block1_conv1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='D_block1_conv2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = ReLU()(x)

        # mid flow
        residual2 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='D_residual2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='D_block2_sepconv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='D_block2_sepconv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='D_add2')([x, residual2])

        residual3 = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='D_residual3', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='D_block3_sepconv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='D_block3_sepconv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='D_add3')([x, residual3])

        residual4 = Conv2D(728, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='D_residual4', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='D_block4_sepconv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='D_block4_sepconv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='D_add4')([x, residual4])

        residual5 = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='D_residual5', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='D_block5_sepconv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(256, (3, 3),strides=(2, 2), padding='same', use_bias=False, name='D_block5_sepconv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = Add(name='D_add5')([x, residual5])

        # exit flow
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='D_block6_sepconv1', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        x = ReLU()(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='D_block6_sepconv2', depthwise_constraint=MaxNorm(5), pointwise_constraint=MaxNorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
        feature = ReLU()(x)

        # dense
        f = Flatten(name='D_flatten')(feature)
        # f = Dropout(0.2)(f)
        f_flatten = Dense(2048, name='D_dense0', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(f)
        f_flatten = ReLU()(f_flatten)
        # f_flatten = Dropout(0.2)(f_flatten)

        # C: orientation indexing
        x = Dense(2048, name='C_dense1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(f_flatten)
        # x = ReLU()(x)
        x = Dense(1024, name='C_dense2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = Dense(256, name='C_dense3', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(x)
        x = Dense(4, name='C_dense4')(x)
        x = Lambda(lambda t: tf.math.l2_normalize(100*t, axis=1), name='normalization')(x)

        # D: real/fake
        y = Dense(1024, name='D_dense1', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(f_flatten)
        y = Activation('relu')(y)
        y = Dense(256, name='D_dense2', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)
        y = Dense(1, name='D_dense3', kernel_constraint=MaxNorm(5), kernel_initializer='he_normal')(y)

        self.D = models.Model(inputs_x, [feature, x, y], name='Discriminator')

    # TODO currently code for training/testing is in train/test.py
    # @tf.function
    # def train_Generator(self):
    #     pass

    # @tf.function
    # def train_Discriminator(self):
    #     pass

    # @tf.function
    # def train(self, dataset):
    #     pass