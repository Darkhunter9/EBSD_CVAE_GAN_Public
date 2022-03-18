"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: imgprocess.py

MODULE: util

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Image processing func


Date:
-------------
2022/03/17 ZD 1.0 public version
"""

from copy import deepcopy
from math import exp, floor, log10, ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image


# Adaptive histogram equalization
def clahe(img, limit=10, tileGridSize=(10, 10)):
    '''
    Contrast Limited Adaptive Histogram Equalization \\
    Refer to https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)

    limit: clipLimit, int, default: 10

    tileGridSize: grid size, (int, int), default: (10, 10)


    Output:
    ------
    img: images after applying clahe, 3darray, shape: (n, h, w)
    '''
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=tileGridSize)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = clahe.apply(img[i])
    return temp

# circular mask
def circularmask(img):
    '''
    Apply largest circular mask

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)


    Output:
    ------
    img: images with circular mask applied, 3darray, shape: (n, h, w)
    '''
    center = [int(img.shape[2]/2), int(img.shape[1]/2)]
    radius = min(center[0], center[1], img.shape[2]-center[0], img.shape[1]-center[1])
    Y, X = np.ogrid[:img.shape[1], :img.shape[2]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask_array = dist_from_center <= radius
    temp = img
    temp[:,~mask_array] = 0
    return temp

# square mask
def squaremask(img):
    '''
    Apply largest square mask inside image with circular mask \\
    Will change the input size

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)


    Output:
    ------
    img: images with square mask applied, 3darray, shape: (n, 0.707*min(h, w), 0.707*min(h, w))
    '''
    n = img.shape[1]
    start = ceil(n*0.5*(1.-0.5**0.5))
    end = floor(n-n*0.5*(1.-0.5**0.5))
    return img[:,start-1:end,start-1:end]

def poisson_noise(img, c=1.):
    '''
    Apply Poisson noise

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)

    c: inverse of noise level (smaller c brings higher noise level), float, default: 1.


    Output:
    ------
    img: images with Poisson noise
    '''
    temp = np.zeros_like(img)

    for i in range(img.shape[0]):
        vals = len(np.unique(img[i]))
        vals = 2 ** np.ceil(np.log2(vals))
        temp[i] = np.random.poisson(img[i] * c * vals) / float(vals) / c
    
    return temp

def bac(img, a=1, b=0):
    '''
    Adjust brightness and contrast

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)

    a: contrast ratio, float, default: 1

    b: brightness offset, float, default: 0


    Output:
    ------
    img: images, 3darray, shape: (n, h, w)
    '''
    temp = np.clip(a*img+b, 0., 255.)
    temp = temp.astype(np.uint8)
    return temp

def gamma_trans(img, gamma):
    '''
    Apply Gamma correction \\
    If gamma < 1:
        The whole figure is brighter, contrast of dark part is increased
        vise versa

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)

    gamma: Gamma value, float


    Output:
    ------
    img: images, 3darray, shape: (n, h, w)
    '''
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = cv2.LUT(img[i], gamma_table)
    temp = temp.astype(np.uint8)
    return temp

def autogamma(img):
    '''
    Apply automatic Gamma correction (gamma = 2.2)

    Input:
    ------
    img: images, 3darray, shape: (n, h, w)


    Output:
    ------
    img: images, 3darray, shape: (n, h, w)
    '''
    meanGrayVal = np.sum(img) / np.prod(img.shape)
    gamma = log10(1/2.2) / log10(meanGrayVal/255.0)
    return gamma_trans(img, gamma)