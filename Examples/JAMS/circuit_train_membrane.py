from __future__ import print_function

__author__ = 'CIRCUIT, June 2017'
__license__ = 'Apache 2.0'


import os, sys, time

import numpy as np

np.random.seed(9999)

import keras
from keras import backend as K

import SimpleITK as sitk

sys.path.append('../..')
from cnn_tools import *
from data_tools import *

if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    tile_size = (256, 256)

    # load raw data
    isbi_dir = os.path.expanduser('~/CIRCUIT/team1/completed/1605')


    train_nii = sitk.ReadImage(os.path.join(isbi_dir, 'team1_waypoint_201610_full_May6_25k_201610_dataset_em_1640.nii.gz'))
    X_train = sitk.GetArrayFromImage(train_nii)
    X_train = X_train.astype(np.float32)
    X_train = X_train / 255.
    X_train = np.expand_dims(X_train, axis=1)


    mem_nii = sitk.ReadImage(os.path.join(isbi_dir, 'team1_waypoint_201610_full_May6_25k_201610_dataset_em_1640_mem.nii.gz'))
    Y_train = sitk.GetArrayFromImage(mem_nii)
    Y_train = Y_train.astype(np.float32)
    Y_train[Y_train==255] = 1
    Y_train = np.expand_dims(Y_train, axis=1)


    valid_nii = sitk.ReadImage(os.path.join(isbi_dir, 'team1_waypoint_201610_full_May6_25k_201610_dataset_em_1750.nii.gz'))
    X_valid = sitk.GetArrayFromImage(valid_nii)
    X_valid= X_valid.astype(np.float32)
    X_valid = X_valid / 255.
    X_valid = np.expand_dims(X_valid, axis=1)


    mem_nii2 = sitk.ReadImage(os.path.join(isbi_dir, 'team1_waypoint_201610_full_May6_25k_201610_dataset_em_1750_mem.nii.gz'))
    Y_valid = sitk.GetArrayFromImage(mem_nii2)
    Y_valid= Y_valid.astype(np.float32)
    Y_valid[Y_valid==255] = 1
    Y_valid = np.expand_dims(Y_valid, axis=1)

    print('[info]: using Keras version:         %s' % str(keras.__version__))
    print('[info]: using backend:               %s' % str(K._BACKEND))
    print('[info]: training data has shape:     %s' % str(X_train.shape))
    print('[info]: training labels has shape:   %s' % str(Y_train.shape))
    print('[info]: validation data has shape:   %s' % str(X_valid.shape))
    print('[info]: validation labels has shape: %s' % str(Y_valid.shape))
    print('[info]: tile size:                   %s' % str(tile_size))

     # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]))
    train_model(X_train, Y_train, X_valid, Y_valid, model,
                n_epochs=12, mb_size=30, n_mb_per_epoch=25)

    print('[info]: total time to train model: %0.2f min' % ((time.time() - tic)/60.))
