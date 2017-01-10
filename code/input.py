import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn_vgg
import utils


def load_image(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(img, (100, 150))
    return resized_img

def load_image_label(path):
    img = skimage.io.imread(path)
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(img, (100, 150))
    return resized_img

def load_batchsize_images(state, batch_size=2):
    imgx_path = 'JPEGImages/'
    imgy_path = 'SegmentationClassAug/'
    x_path_batch = []
    y_path_batch = []
    x_batch = []
    y_batch = []
    file_path = ''
    if state == 'train':
        file_path = 'fcn_train.txt'
    elif state == 'validation':
        file_path = 'fcn_val.txt'
    else:
        file_path = 'test_fcn.txt'

    with open(file_path) as f:
        lines = random.sample(f.readlines(),batch_size)
        for line in lines:
            line = line.split()
            x_path_batch.append(imgx_path+line[0])
            y_path_batch.append(imgy_path+line[1])
    f.close()
        
    for x_path in x_path_batch:
        x_batch.append(load_image(x_path))
    for y_path in y_path_batch:
        y_batch.append(load_image_label(y_path))

    return np.asarray(x_batch), np.asarray(y_batch)