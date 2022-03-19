"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: loss.py

MODULE: CVAE_GAN

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Loss functions for EBSD_CVAE_GAN model.

Date:
-------------
2022/03/18 ZD 1.0 public version
"""

import os
import sys
import time
import numpy as np
from math import pi

import tensorflow as tf

from tensorflow.keras.backend import epsilon

# *********************************************
# Loss of Generator
# *********************************************
'''
For VAE, loss ELBO = E_(q(z|x))[log(p(x,z)/q(z|x))]
For z sampled from q(z|x): ELBO -> log(p(x|z)) + log(p(z)) - log(q(z|x))
'''
@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
    '''
    pdf of log-normal distribution

    Input:
    ------
    sample: sampled values, ndtensor

    mean: mean of distribution, ndtensor

    logvar: log of variance, ndtensor

    raxis: axis to reduce, int, default: 1


    Output:
    ------
    pdf: reduced mean pdf, (n-1)dtensor
    '''
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_mean(tf.reduce_sum(
      .5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis))

@tf.function
def Crossentropy(y_true, y_pred):
    '''
    logp(x|z)

    Input:
    ------
    y_true: ground truth, ndtensor

    y_pred: prediction, ndtensor


    Output:
    ------
    crossentropy: reduced mean crossentropy, 0dtensor (float)
    '''
    cross_ent = -(y_true*tf.math.log(y_pred+epsilon()) + (1-y_true)*tf.math.log(1-y_pred+epsilon()))
    l = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    return l

@tf.function
def featureMatchingLoss(y_true, y_pred):
    '''
    MSE of extracted features

    Input:
    ------
    y_true: ground truth, ndtensor

    y_pred: prediction, ndtensor


    Output:
    ------
    mse: reduced mean MSE, 0dtensor (float)
    '''
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3]))

@tf.function
def KL(mean, logvar, mode='a'):
    '''
    KL divergence

    Input:
    ------
    mean: mean of distribution, ndtensor

    logvar: log of variance, ndtensor

    mode:
        'a': analytical
        'mc': Monte Carlo


    Output:
    ------
    kl: reduced mean kl divergence, 0dtensor (float)
    '''
    if mode == 'mc':
        z = tf.random.normal(shape=mean.shape) * tf.exp(logvar*.5) + mean
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return logpz - logqz_x
    else:
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(logvar) + tf.square(mean) -1. - logvar, axis=1))


# *********************************************
# Loss of Discriminator
# *********************************************
'''
The use of acos will easily make loss 0 as its derivative has value restrictions.
'''
def loss_qu(y_true, y_pred):
    '''
    calculating 'distance' between two quaternions
    1 - <y_true, y_pred> = cos(theta/2)
    not considering the rotation axis

    Input:
    ------
    y_true: ground truth orientation in form of quaternion, 2dtensor, shape: (n, 4)

    y_pred: prediction orientation in form of quaternion, 2dtensor, shape: (n, 4)


    Output:
    ------
    misorientation: mean misorientation, 0dtensor (float)
    '''
    loss = 1 - tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1))
    # loss = 360./pi*K.mean(
    #     tf.math.acos(
    #         tf.clip_by_value(
    #             tf.reshape(K.batch_dot(y_true, y_pred, axes=1), shape=[-1,1]),
    #         clip_value_min=-1, clip_value_max=1)))

    return 10000*loss


def loss_disorientation(y_true, y_pred):
    '''
    only works for cubic material
    point group higher than 432

    calculating disorientation angle between two quaternions
    not considering the rotation axis

    Input:
    ------
    y_true: ground truth orientation in form of quaternion, 2dtensor, shape: (n, 4)

    y_pred: prediction orientation in form of quaternion, 2dtensor, shape: (n, 4)


    Output:
    ------
    disorientation: mean disorientation (in unit of rad), 0dtensor (float)
    '''
    y_pred_star = tf.multiply(y_pred, tf.constant([[1.,-1.,-1.,-1.]]))

    temp = tf.reshape(tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1), shape=[-1,1]) # w of misorientation
    temp2 = tf.multiply(y_pred_star[:,1:], y_true[:,0:1]) + tf.multiply(y_true[:,1:], y_pred_star[:,0:1]) + tf.linalg.cross(y_true[:,1:], y_pred_star[:,1:]) # x y z of misorientation
    misorientation = tf.sort(tf.abs(tf.concat([temp,temp2], axis=-1)), axis=-1, direction='DESCENDING') # sort w >= x >= y >= z >= 0

    temp3 = tf.concat((misorientation[:,0:1], # w0 == w
                    tf.reshape(0.5*tf.reduce_sum(misorientation,axis=-1), shape=[-1,1]), # w1 == (w+x+y+z)/2
                    tf.reshape(0.5**0.5*tf.reduce_sum(misorientation[:,0:2], axis=-1), shape=[-1,1])), axis=-1) # w2 == (w+x) * sqrt（2）/ 2

    # the mean disorientation angle is:
    return 36000./pi * tf.reduce_mean(tf.acos(tf.clip_by_value(tf.reduce_max(temp3, axis=-1), -1.0, 1.0))) # average(degrees(2*acos(max(w0, w1, w2))))


def D_loss(r_logit, f_logit):
    '''
    For Discriminator:
    L = E_(x~Pr)[f_w(x)] - E_(x~Pg)[f_w(x)]
    Discriminator tries to maximize this term (i.e. minimize -L)

    Input:
    ------
    r_logit: log of points for real images, ndtensor

    f_logit: log of points for fake images, ndtensor

    Output:
    ------
    -L: difference between points for real and fake images, 0dtensor (float)
    '''
    # r_loss = - tf.reduce_mean(r_logit)
    # f_loss = tf.reduce_mean(f_logit)
    r_loss = - tf.reduce_mean(tf.math.sigmoid(r_logit))
    f_loss = tf.reduce_mean(tf.math.sigmoid(f_logit))
    return r_loss + f_loss


def G_loss(f_logit):
    '''
    For Generator:
    L = E_(x~Pg)[f_w(x)]
    Generator tries to maximize this term (i.e. minimize -L).

    Input:
    ------
    f_logit: log of points for fake images, ndtensor

    Output:
    ------
    -L: negative points for fake images, 0dtensor (float)
    '''
    # f_loss = - tf.reduce_mean(f_logit)
    f_loss = - tf.reduce_mean(tf.math.sigmoid(f_logit))
    return f_loss