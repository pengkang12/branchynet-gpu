#!/usr/bin/env python -W ignore::DeprecationWarning

# -*- coding: utf-8 -*-
"""
Created on Dec 1, 2021
author: Peng Kang

"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import chainer.functions as F
import chainer.links as L
from chainer import cuda
from sklearn.datasets import fetch_openml
from tensorflow.keras import models

import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import mnist
from networks import lenet_mnist
from branchynet.links import *
from branchynet import utils, visualize
from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda
import dill
import psutil
import time

#define how many times we measure performance, and get the average result. 
REPEAT = 3
model_name="../_models/lenet_fashion_mnist.bn"
autoencoder_name='../_models/autoencoder/autoencoder_fmnist.h5'
threshold_base=0.5
TEST_BATCHSIZE = 10    
USE_GPU = False

def measure_performance_LeNet(X, Y):
    # load branchynet
    branchyNet = None
    with open(model_name, "rb") as f:
        branchyNet = dill.load(f)
    #set network to inference mode, this is fob_test_data_yr measuring baseline function. 
    branchyNet.testing()
    branchyNet.verbose = False

    #branchyNet.to_cpu()
    cpu_time_a = (time.time(), psutil.cpu_times())

    if USE_GPU and cuda.available:
        branchyNet.to_gpu()

    res_basediff = []
    for i in range(REPEAT):
        c_baseacc, c_basediff, _, _ = utils.test(branchyNet, X, Y, main=True, batchsize=TEST_BATCHSIZE)
        res_basediff.append(c_basediff)

    print("LeNet accuracy is ", c_baseacc)
    print("LeNet time is ", sum(res_basediff)/REPEAT)
    print("\n")
    cpu_time_b = (time.time(), psutil.cpu_times())
    print 'CPU used in %0.2f seconds: %s' % (
        cpu_time_b[0] - cpu_time_a[0],
        calculate(cpu_time_a[1], cpu_time_b[1])
    )
    return c_baseacc, c_basediff


def measure_performance_branchynet(X, Y,threshold=0):
    # load branchynet
    branchyNet = None
    with open(model_name, "rb") as f:
        branchyNet = dill.load(f)
    #set network to inference mode, this is fob_test_data_yr measuring baseline function. 
    branchyNet.testing()
    branchyNet.verbose = False

    #branchyNet.to_cpu()
    thresholds = [threshold_base+threshold]
    #print(decoded_imgs.shape)
    
    cpu_time_a = (time.time(), psutil.cpu_times())
    if USE_GPU and cuda.available:
        branchyNet.to_gpu()
        pass
    res_diff = [] 
    for i in range(REPEAT):
    	c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, X, Y, thresholds,
                                                       batchsize=TEST_BATCHSIZE, verbose=False)
        c_diffs *= len(Y)
        res_diff.append(c_diffs)

    print("accuracy is ", c_accs)
    print("branchyNet time is ", sum(res_diff)/REPEAT)
    print("the distribution of exit number is ", c_exits)
    
    cpu_time_b = (time.time(), psutil.cpu_times())
    print 'CPU used in %d seconds: %s' % (
        cpu_time_b[0] - cpu_time_a[0],
        calculate(cpu_time_a[1], cpu_time_b[1])
    )
    return c_accs, c_diffs
# show some easy data and hard data


def measure_perf_and_time(X_test_tmp, Y_test_tmp, data_tmp_shape, threshold=0.5):
    
    b_test_data_x = X_test_tmp.reshape(data_tmp_shape)
    autoencoder = models.load_model(autoencoder_name)

    cpu_time_a = (time.time(), psutil.cpu_times())
    
    start = time.time()
    for i in range(REPEAT):
        decoded_imgs = autoencoder.predict(b_test_data_x)
    end = time.time()
    
    cpu_time_b = (time.time(), psutil.cpu_times())

    autoencoder_cpu = calculate(cpu_time_a[1], cpu_time_b[1])
    autoencoder_time = (cpu_time_b[0] - cpu_time_a[0])/REPEAT
    print('autoencoder CPU used in %.2f %.2f seconds: %s' % (
        (end-start)/REPEAT,
        autoencoder_time,
        autoencoder_cpu ))
 
    acc, diff = measure_performance_branchynet(decoded_imgs.reshape(-1, 1, 28,28)*255.0, Y_test_tmp, threshold=threshold)
    print("branchynet total time(s) is ", diff)

    print("\n")
    return acc, autoencoder_time + diff


def calculate(t1, t2):
    # from psutil.cpu_percent()
    # see: https://github.com/giampaolo/psutil/blob/master/psutil/__init__.py
    t1_all = sum(t1)
    t1_busy = t1_all - t1.idle
    t2_all = sum(t2)
    t2_busy = t2_all - t2.idle
    if t2_busy <= t1_busy:
        return 0.0
    busy_delta = t2_busy - t1_busy
    all_delta = t2_all - t1_all
    busy_perc = (busy_delta / all_delta) * 100
    return round(busy_perc, 1)

import tensorflow as tf
# Load the fashion-mnist pre-shuffled train data and test data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", X_train.shape, "y_train shape:", Y_train.shape)
X_train, X_test = X_train , X_test
#print(Y_test.shape)
#np.savetxt('Fashion_X_or.txt',X_test.reshape(10000,784))
#np.savetxt('Fashion_Y_or.txt',Y_test.reshape(10000,))
X_check_test= X_test.reshape(10000,784)
X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28) / 255.0


def measure_perf():
    print("\nmeasure branchyNet")
    measure_performance_branchynet(X_test*255.0, Y_test)
    print("\n\nmeasure BranchyNet with early exit")
    measure_performance_branchynet(X_test*255.0, Y_test, 2)
    print("\n\nmeasure LeNet")
    measure_performance_LeNet(X_test*255.0, Y_test)
    print("\n\nmeasure performance data,all data go into different exits")
    measure_perf_and_time(X_test, Y_test, (-1, 784), 2)


def scale_analysis(percentile=0.1):

    X_test_part = []
    Y_test_part = []

    for i in range(10):
        X_test_part.append(X_test[Y_test == i][0:int(1000*percentile), :])
        Y_test_part.append(Y_test[Y_test == i][0:int(1000*percentile)])

    X_test_part = np.concatenate(X_test_part)
    Y_test_part = np.concatenate(Y_test_part).reshape(-1, )

    acc, diff = measure_performance_branchynet(X_test_part*255, Y_test_part)
    print("running time(s) is ", diff)
    return acc,  diff

acc = []
run_time = []
for i in range(1, 11, 1):
    _acc, _time = scale_analysis(i/10.0)
    acc.append(_acc)
    run_time.append(_time)
print(list(map(lambda x: x[0], acc)))
print(list(map(lambda x: x[0], run_time)))


def scale_analysis(percentile=0.1):

    X_test_part = []
    Y_test_part = []


    for i in range(10):
        X_test_part.append(X_test[Y_test == i][0:int(1000*percentile), :])
        Y_test_part.append(Y_test[Y_test == i][0:int(1000*percentile)])
    
    X_test_part = np.concatenate(X_test_part)
    Y_test_part = np.concatenate(Y_test_part).reshape(-1, )

    return measure_perf_and_time(X_test_part, Y_test_part, (-1, 784), 2)
    
acc = []
run_time = []
for i in range(1, 11, 1):
    _acc, _time = scale_analysis(i/10.0)
    acc.append(_acc)
    run_time.append(_time)
print(list(map(lambda x: x[0], acc)))
print(list(map(lambda x: x[0], run_time)))
