"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: training_CVAE.py

MODULE: CVAE_GAN

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Training EBSD_CVAE models.
Consider both reconstruction error and latent variable error.

Date:
-------------
2022/03/21 ZD 1.0 public version
"""

import os
import sys
import time
import datetime
import numpy as np
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import save_model, load_model

from models import EBSD_GAN
from loss import *
from utils.data_generator import DataGenerator


# *********************************************
# 0. set GPUs
# *********************************************
# assign gpu to use
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Enabling device placement logging causes any Tensor allocations or operations to be printed
# tf.debugging.set_log_device_placement(True)

# set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# fix random seed for reproducibility
# seed = 7
seed = int(time.time())
np.random.seed(seed)


# *********************************************
# 1. generate model and data
# *********************************************
params = {  # hyperparam
            'dim': (60,60), # EBSD pattern shape: (h, w)
            'batch_size': 16, # batch size
            'n_channels': (1,), # number of channels
            'n_epochs': 30, # training epochs
            'n_d': 9, # frequency of training discriminator 
                      # (here only used to output fake patterns during training)
            'n_latent': 500, # latent representation dimension
            'n_att': 4, # attribute dimension

            # G
            # losses = loss_rec +
            #          params['g_loss_weight_rec_p'] * loss_rec_p +
            #          params['g_loss_weight_latent'] * loss_latent * tf.case([(loss_latent > 15., lambda: 1.5),
            #                                                                (loss_latent > 5., lambda: 0.1),
            #                                                                ], default=lambda: 0.01)
            'g_loss_weight_rec_p': 0.,
            'g_loss_weight_latent': 0.8,
            'lr_G': 8*10**(-6),

            # directory
            'data': 'dir/to/training_dataset.h5',
            'tmp': 'dir/to/generated_fake_patterns',
            'saved_model': 'dir/to/saved_model.h5',
            'loaded_model': 'dir/to/pre_trained_model.h5',
            'csv': 'dir/to/training_log.csv',}

model = EBSD_GAN(img_size=tuple(list(params['dim'])+list(params['n_channels'])),
                    batch_size=params['batch_size'],
                    latent_size=params['n_latent'],
                    att_size=params['n_att'])

imgprocesser = ['clahe(10, (4, 4))']
training_generator = DataGenerator(data=params['data'],
                                    batch_size=params['batch_size'],
                                    dim=params['dim'],
                                    n_channels=params['n_channels'],
                                    shuffle=True,
                                    processing=imgprocesser)

# load model weights
if params['loaded_model']:
    model.G.load_weights(params['loaded_model'])
    print('model weights loaded.')

# model architecture output for debug
# model.Enc.summary()
# model.Dec.summary()
# model.G.summary()

# optimizer
opt_G = tf.keras.optimizers.Adam(beta_1=0.5)


# *********************************************
# 2. define training
# *********************************************
def train_G(model, patterns, orientations, opt, lr):
    orientations = tf.cast(orientations, tf.float32)
    patterns = tf.cast(patterns, tf.float32)

    with tf.GradientTape() as tape:
        # generate
        # z from VAE input
        mean, logvar, patterns_fake = model.G([patterns, orientations])
        # z from normal distribution
        patterns_fake_p = model.sample(orientations)
        
        # generator losses
        loss_rec = Crossentropy(patterns, patterns_fake)
        loss_rec_p = Crossentropy(patterns, patterns_fake_p)
        loss_latent = KL(mean, logvar)
        
        losses = loss_rec + params['g_loss_weight_rec_p'] * loss_rec_p + params['g_loss_weight_latent'] * loss_latent * tf.case([
                                                                                                                                (loss_latent > 15., lambda: 1.5),
                                                                                                                                (loss_latent > 5., lambda: 0.1),
                                                                                                                                ], default=lambda: 0.01)

    grads = tape.gradient(losses, model.G.trainable_variables)
    print("G_loss, loss_rec:{:.3f}, loss_rec_p:{:.3f}, loss_z:{:.3f}".format(loss_rec.numpy(), loss_rec_p.numpy(), loss_latent.numpy()))

    # optim
    # update learning rate
    opt.learning_rate = lr
    opt.apply_gradients(zip(grads, model.G.trainable_variables))

    # return losses
    return [loss_rec, loss_rec_p, loss_latent]


# *********************************************
# 3. training
# *********************************************
step = 0
fig = plt.figure(figsize=(4,12))
loss_G_list = []
with open(params['csv'], 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "lr", "loss_rec", "loss_rec_p", "loss_latent", "loss_G", "time"])

for ep in range(params['n_epochs']):
    print('epoch: {:d}'.format(ep))
    time_ep = time.time()
    loss_G = [0.] * 3

    for it, (patterns, orientations) in zip(range(len(training_generator)), training_generator):
        print('epoch: {:d}, iteration: {:d}/{:d}, lr: {:e}, tr: {}'.format(ep, it+1, len(training_generator), params['lr_G'], 
                                                                            str(datetime.timedelta(seconds=(time.time()-time_ep)/(it+1)*(len(training_generator)-it-1)))[:8]))
        step += 1

        # training for single step
        losses = train_G(model, patterns, orientations, opt_G, params['lr_G'])
        for i in range(len(loss_G)):
            loss_G[i] += losses[i]

        # output fake patterns every 100 * (params['n_d'] + 1) steps
        if step % (100 * (params['n_d'] + 1)) == 0:
            orientations = orientations[:min(16, len(orientations))]
            patterns_fake_1 = model.sample(orientations).numpy()

            patterns_fake_2 = model.G([patterns[:min(16, len(patterns))], orientations[:min(16, len(orientations))]], training=False)[-1].numpy()

            plt.cla()

            # plot generated patterns from sampling
            for i in range(min(16, patterns_fake_1.shape[0])):
                plt.subplot(12, 4, i+1)
                plt.imshow(patterns_fake_1[i, :, :, 0], cmap='gray')
                plt.axis('off')

            # plot generated patterns with patterns input
            for i in range(min(16, patterns_fake_2.shape[0])):
                plt.subplot(12, 4, i+17)
                plt.imshow(patterns_fake_2[i, :, :, 0], cmap='gray')
                plt.axis('off')
            
            # plot origin patterns
            for i in range(min(16, patterns.shape[0])):
                plt.subplot(12, 4, i+33)
                plt.imshow(patterns[i, :, :, 0], cmap='gray')
                plt.axis('off')
            
            plt.savefig(os.path.join(params['tmp'], 'ep_{:03d}_it_{:05d}.png'.format(ep, it)))

    # log
    for i in range(len(loss_G)):
        loss_G[i] /= len(training_generator)
    loss_G.append(tf.reduce_sum(loss_G))
    loss_G_list.append([i.numpy() for i in loss_G])

    # callbacks at the end of each epoch
    # csv
    with open(params['csv'], 'a') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([ep, params['lr_G']] + loss_G_list[-1] + [str(datetime.timedelta(seconds=time.time()-time_ep))])

    # model saving
    if len(loss_G_list) > 1 and loss_G_list[-1][-1] < loss_G_list[-2][-1]:
        model.G.save_weights(params['saved_model'], save_format='h5')
        print('model saved to ' + params['saved_model'])

    # shuffle training data
    training_generator.on_epoch_end()

    print('\n' * 10)