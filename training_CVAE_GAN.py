"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: training_CVAE_GAN.py

MODULE: CVAE_GAN

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Training EBSD_CVAE_GAN models.
Both CVAE and discriminator in GAN are pre-trained.

Date:
-------------
2022/03/21 ZD 1.0 public version
"""

import os
import sys
import math
import time
import datetime
import itertools
import numpy as np
from tqdm import tqdm, trange
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
from utils.eu2qu import eu2qu
from utils.imgprocess import *


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
            'batch_size': 8, # batch size
            'n_channels': (1,), # number of channels
            'n_epochs': 30, # training epochs
            'n_d': 10, # frequency of training discriminator 
                       # (here only used to output fake patterns during training)
            'n_latent': 500, # latent representation dimension
            'n_att': 4, # attribute dimension

            # D
            # losses = loss_o + params['d_loss_weight_d'] * loss_d
            'd_loss_weight_d': 1.,
            'lr_D': 4*10**(-6),

            # G
            # losses = loss_rec +
            #          params['g_loss_weight_rec_p'] * loss_rec_p +
            #          params['g_loss_weight_latent'] * loss_latent * tf.case([(loss_latent > 15., lambda: 1.5),
            #                                                                (loss_latent > 5., lambda: 0.1),
            #                                                                ], default=lambda: 0.01)
            #          params['g_loss_weight_o'] * loss_o +
            #          params['g_loss_weight_d'] * loss_d
            'g_loss_weight_rec_p': 0.,
            'g_loss_weight_latent': 0.8,
            'g_loss_weight_o': 0.1,
            'g_loss_weight_d': 0.05,
            'lr_G': 4*10**(-6),

            # directory
            'data': 'dir/to/training_dataset.h5',
            'tmp': 'dir/to/generated_fake_patterns',
            'saved_model_G': 'dir/to/saved_model_G.h5',
            'saved_model_D': 'dir/to/saved_model_D.h5',
            'loaded_model_G': 'dir/to/pre_trained_model_G.h5',
            'loaded_model_D': 'dir/to/pre_trained_model_D.h5',
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
if params['loaded_model_G']:
    model.G.load_weights(params['loaded_model_G'])
    print('model G weights loaded.')
if params['loaded_model_D']:
    model.D.load_weights(params['loaded_model_D'])
    print('model D weights loaded.')

# model architecture output for debug
# model.Enc.summary()
# model.Dec.summary()
# model.G.summary()
# model.D.summary()

# define trainable parts in discriminator
layers = ['D_dense1', 'D_dense2', 'D_dense3']
D_trainable_variables = [model.D.get_layer(i).trainable_weights for i in layers]
D_trainable_variables = list(itertools.chain(*D_trainable_variables))

# optimizer
opt_G = tf.keras.optimizers.Adam(beta_1=0.5)
opt_D = tf.keras.optimizers.Adam(beta_1=0.5)


# *********************************************
# 2. define training
# *********************************************
def train_D(model, patterns, orientations, opt, lr):
    orientations = tf.cast(orientations, tf.float32)
    patterns = tf.cast(patterns, tf.float32)

    patterns_fake = model.G([patterns, orientations])[-1]
    patterns_fake_p = model.sample(orientations)

    with tf.GradientTape() as tape:
        real_f, real_o, real_gan = model.D(patterns, training=True)
        fake_f, fake_o, fake_gan = model.D(patterns_fake, training=True)
        fake_f_p, fake_o_p, fake_gan_p = model.D(patterns_fake_p, training=True)

        # discriminator losses
        loss_o = loss_qu(orientations, real_o)
        loss_d = tf.reduce_mean(tf.math.sigmoid(tf.concat([fake_gan_p, -real_gan], axis=0)))
        losses = loss_o + params['d_loss_weight_d'] * loss_d

    # freeze weights of classifier and feature extractor, only train discriminator
    grads = tape.gradient(losses, D_trainable_variables)
    print("D_loss, loss_o:{:.3f}, loss_d:{:.3f}".format(loss_o.numpy(), loss_d.numpy()))

    # optim
    # update learning rate
    opt.learning_rate = lr
    opt.apply_gradients(zip(grads, D_trainable_variables))

    return [loss_o, loss_d]

def train_G(model, patterns, orientations, opt, lr):
    orientations = tf.cast(orientations, tf.float32)
    patterns = tf.cast(patterns, tf.float32)

    with tf.GradientTape() as tape:
        # generate
        # z from VAE input
        mean, logvar, patterns_fake = model.G([patterns, orientations])
        # z from normal distribution
        patterns_fake_p = model.sample(orientations)

        # discriminate
        real_f, real_o, real_gan = model.D(patterns, training=True)
        fake_f, fake_o, fake_gan = model.D(patterns_fake, training=True)
        fake_f_p, fake_o_p, fake_gan_p = model.D(patterns_fake_p, training=True)

        # generator losses
        # loss from generator
        loss_rec = Crossentropy(patterns, patterns_fake)
        loss_rec_p = Crossentropy(patterns, patterns_fake_p)
        loss_latent = KL(mean, logvar)
        # loss from distriminator
        # orientation
        loss_o = loss_qu(real_o, fake_o_p)
        # real/fake
        loss_d = tf.reduce_mean(tf.math.sigmoid(-fake_gan_p))

        losses = (loss_rec +
                    params['g_loss_weight_rec_p'] * loss_rec_p +
                    params['g_loss_weight_latent'] * loss_latent * tf.case([(loss_latent > 15., lambda: 1.5),
                                                                            (loss_latent > 5., lambda: 0.1),
                                                                            ], default=lambda: 0.01) + 
                    params['g_loss_weight_o'] * loss_o +
                    params['g_loss_weight_d'] * loss_d)

    grads = tape.gradient(losses, model.Dec.trainable_variables)
    print("G_loss, loss_rec:{:.3f}, loss_rec_p:{:.3f}, loss_z:{:.3f}, loss_o:{:.3f}, loss_d:{:.3f}".format(loss_rec.numpy(),
                                                                                            loss_rec_p.numpy(),
                                                                                            loss_latent.numpy(),
                                                                                            loss_o.numpy(),
                                                                                            loss_d.numpy()))

    # optim
    # update learning rate
    opt.learning_rate = lr
    opt.apply_gradients(zip(grads, model.Dec.trainable_variables))

    # return losses
    return [loss_rec, loss_rec_p, loss_latent, loss_o, loss_d]


# *********************************************
# 3. training
# *********************************************
step = 0
loss_G_list = []
loss_D_list = []
step_G = 0
step_D = 0

# initialize orientations for test
idx = [0, 5, 10, 11]
orientations = training_generator.y[idx,:]
orientations = orientations.astype(float)
orientations_temp = np.zeros((len(orientations),4))
for i in range(len(orientations)):
    orientations_temp[i] = eu2qu(orientations[i])
orientations_temp = np.clip(np.nan_to_num(orientations_temp),-1.,1.)
orientations_test = orientations_temp
print('orientations for test')
print(orientations_test)
# initialize patterns for comparison
patterns_test = training_generator.X[idx]
patterns_test = patterns_test.astype(np.uint8)
for i in imgprocesser:
    patterns_test = eval(i.replace('(','(patterns_test,',1))
patterns_test = patterns_test.astype(float)
patterns_test = np.clip(np.nan_to_num(patterns_test),0.,255.) / 255.0
patterns_test = patterns_test[:,:,:,np.newaxis]
# initialize matplotlib
fig, axs = plt.subplots(3, len(idx), figsize=(len(idx),3))
for i in range(len(idx)):
    axs[-1][i].clear()
    axs[-1][i].imshow(patterns_test[i,:,:,0], cmap='gray')
    axs[-1][i].axis('off')

with open(params['csv'], 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "lr_G", "loss_rec", "loss_rec_p", "loss_latent", "loss_orientation", "loss_dis", "loss_G",
                    "lr_D", "loss_orientation", "loss_dis", "loss_D", "time"])

for ep in range(params['n_epochs']):
    print('epoch: {:d}'.format(ep))
    time_ep = time.time()
    loss_G = [0.] * 5
    loss_D = [0.] * 2
    step_D = 0
    step_G = 0

    for it, (patterns, orientations) in zip(range(len(training_generator)), training_generator):
        print('epoch: {:d}, iteration: {:d}/{:d}, lr_G: {:e}, lr_D: {:e}, tr: {}'.format(ep, it+1, len(training_generator), params['lr_G'], params['lr_D'],
                                                                            str(datetime.timedelta(seconds=(time.time()-time_ep)/(it+1)*(len(training_generator)-it-1)))[:8]))
        step += 1
        
        # train D
        if not step % (params['n_d'] + 1):
            step_D += 1
            losses = train_D(model, patterns, orientations, opt_D, params['lr_D'])
            for i in range(len(loss_D)):
                loss_D[i] += losses[i]

        # train G
        else:
            step_G += 1
            losses = train_G(model, patterns, orientations, opt_G, params['lr_G'])
            for i in range(len(loss_G)):
                loss_G[i] += losses[i]

        # output fake patterns every 100 * (params['n_d'] + 1) steps
        if step % (100 * (params['n_d'] + 1)) == 0:
            patterns_fake_1 = model.sample(orientations_test).numpy()

            patterns_fake_2 = model.G([patterns_test, orientations_test], training=False)[-1].numpy()

            for i in range(len(idx)):
                # plot generated patterns from sampling
                axs[0][i].clear()
                axs[0][i].imshow(patterns_fake_1[i,:,:,0], cmap='gray')
                axs[0][i].axis('off')
                # plot generated patterns with patterns input
                axs[1][i].clear()
                axs[1][i].imshow(patterns_fake_2[i,:,:,0], cmap='gray')
                axs[1][i].axis('off')
            
            plt.savefig(os.path.join(params['tmp'], 'ep_{:03d}_it_{:05d}.png'.format(ep, it)))
        
    if step_D:
        for i in range(len(loss_D)):
            loss_D[i] /= step_D
    if step_G:
        for i in range(len(loss_G)):
            loss_G[i] /= step_G

    # log
    loss_D.append(tf.reduce_sum(loss_D))
    loss_G.append(tf.reduce_sum(loss_G))
    loss_D_list.append([i.numpy() for i in loss_D])
    loss_G_list.append([i.numpy() for i in loss_G])

    # callbacks at the end of each epoch
    # csv
    with open(params['csv'], 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ep, params['lr_G']] + loss_G_list[-1] + [params['lr_D'],] + loss_D_list[-1] + [str(datetime.timedelta(seconds=time.time()-time_ep))])

    # model saving
    if len(loss_G_list) > 1 and math.isclose(loss_G_list[-1][-1], min([i[-1] for i in loss_G_list]), abs_tol=0.2) and params['saved_model_G']:
        model.G.save_weights(params['saved_model_G'], save_format='h5')
        print('model G saved to ' + params['saved_model_G'])
    if len(loss_D_list) > 1 and math.isclose(loss_D_list[-1][-1], min([i[-1] for i in loss_D_list]), abs_tol=0.2) and params['saved_model_D']:
        model.D.save_weights(params['saved_model_D'], save_format='h5')
        print('model D saved to ' + params['saved_model_D'])

    # shuffle training data
    training_generator.on_epoch_end()

    print('\n' * 10)