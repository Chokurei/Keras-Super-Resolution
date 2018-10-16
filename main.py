#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@CreateTime:   2018-10-10T12:16:31+09:00
@Email:  guozhilingty@gmail.com
@Copyright: Chokurei
@License: MIT
"""
from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import cv2
import pandas as pd

from prepare_data import prepare_train_data, write_hdf5, prepare_test_data
import argparse


def psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

def model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, 9, activation='relu', padding='valid', input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(64, 3, activation='relu', padding='same'))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(1, 5, activation='linear', padding='valid'))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, 9, activation='relu', padding='valid', input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(64, 3, activation='relu', padding='same'))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(1, 5, activation='linear', padding='valid'))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train(data, model_path, epochs):

    srcnn_model = model()
    print(srcnn_model.summary())
#    data, label = pd.read_training_data("./logs/data_cache/train_data.h5")
#    val_data, val_label = pd.read_training_data("./logs/data_cache/test_data.h5")
    X_train, y_train, X_val, y_val = data
    X_train, X_val = np.transpose(X_train,(0,2,3,1)), np.transpose(X_val,(0,2,3,1))
    y_train, y_val = np.transpose(y_train,(0,2,3,1)), np.transpose(y_val,(0,2,3,1))
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(X_train, y_train, batch_size=128, validation_data=(X_val, y_val),
                    callbacks=callbacks_list, shuffle=True, epochs=epochs, verbose=0)

def predict(model, img_path, result_path):
    srcnn_model = predict_model()
    srcnn_model.load_weights(model)
#    srcnn_model.load_weights("SRCNN_check_building.h5")
#    IMG_NAME = "/home/mark/Engineer/SR/data/Set14/flowers.bmp"
    img_names = os.listdir(img_path)
    names = []
    bicubic = []
    SRCNN = []

    for img_name in img_names:
        name, form = os.path.splitext(img_name)
#        hr_img_name = name + '_hr' + form
        lr_img_name = name + '_lr' + form
        sr_img_name = name + '_sr' + form
            
        img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_COLOR)
        img_hr = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        shape = img.shape
        Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
        img[:, :, 0] = Y_img
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(result_path,lr_img_name), img)
        img_lr = np.copy(img)
    
        Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = Y_img.astype(float) / 255.
        pre = srcnn_model.predict(Y, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(result_path,sr_img_name), img)
        img_sr = np.copy(img)

        # psnr calculation:
        im1 = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_COLOR)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
        im2 = cv2.imread(os.path.join(result_path,lr_img_name), cv2.IMREAD_COLOR)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
        im3 = cv2.imread(os.path.join(result_path,sr_img_name), cv2.IMREAD_COLOR)
        im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

        imgs = [img_hr, img_lr, img_sr]
        result_img_compare_save(imgs, os.path.join(result_path, name+'.png'))
        
        names.append(name)
        bicubic.append(cv2.PSNR(im1, im2))
        SRCNN.append(cv2.PSNR(im1, im3))
        stats = [names, bicubic, SRCNN]
    return stats

def result_img_compare_save(images, path):
    titles = ['high resolution', 'low resolution', 'super resolution']
    plt.figure(figsize=(10, 4))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.title(titles[i])
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.savefig(path)


def result_stats_save(stats, path):
    result = pd.DataFrame(stats).T
    result = result.rename(columns = {0:'name',1:'bicubic',2:'SRCNN'})
    result.to_csv(os.path.join(path,'stats.csv'))
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Keras Super Res Example')
    parser.add_argument('--train_data_path', type=str, default= "./data/train/", help='training data path')
    parser.add_argument('--test_data_path', type=str, default= "./data/test/", help='testing data path')

    parser.add_argument('--model_path', type=str, default='./logs/model_zoo/', help='model path')
    parser.add_argument('--model_name_train', type=str, default="guo.h5", help='trained model name')
    parser.add_argument('--model_name_predict', type=str, default="guo.h5", help='model used to predict')

    parser.add_argument('--result_path', type=str, default="./result", help='model path')
    parser.add_argument('--result_stats_path', type=str, default="./logs/statistic/", help='trained model name')

    parser.add_argument('-t','--train_mode', type=lambda x: (str(x).lower() == 'true'), default=True, help='train the model or not')
    parser.add_argument('-i','--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('-u','--upscale_factor', type=int, default=2, help="super resolution upscale factor")
   
    opt = parser.parse_args()
    
    if opt.train_mode:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('===> Loading datasets')
        train_data, train_label = prepare_train_data(opt.train_data_path, opt.upscale_factor)
        print(train_data.shape)
        print(train_label.shape)
        test_data, test_label = prepare_test_data(opt.test_data_path, opt.upscale_factor)
        print(test_data.shape)
        print(test_label.shape)
        data_all = [train_data, train_label, test_data, test_label]
        print('===> Building model')
        train(data_all, os.path.join(opt.model_path, opt.model_name_train), opt.nEpochs)
        model_name_predict = opt.model_name_train
        print('===> Testing')
        stats = predict(os.path.join(opt.model_path, model_name_predict), opt.test_data_path, opt.result_path)
    else:
        print('===> Testing')
        stats = predict(os.path.join(opt.model_path, opt.model_name_predict), opt.test_data_path, opt.result_path)
    result_stats_save(stats, opt.result_stats_path)
    print('===> Complete')

