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
from datasets.qmnist import qmnist
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
model_name="../_models/lenet_mnist.bn"
autoencoder_name='../_models/autoencoder/autoencoder_lenet.h5'
threshold_base=0.05
TEST_BATCHSIZE = 10    

def measure_performance_LeNet(X, Y):
    # load branchynet
    branchyNet = None
    with open(model_name, "rb") as f:
        branchyNet = dill.load(f)
    #set network to inference mode, this is fob_test_data_yr measuring baseline function. 
    branchyNet.testing()
    branchyNet.verbose = False

    #branchyNet.to_cpu()
    
    if cuda.available:
        branchyNet.to_gpu()

    res_basediff = []
    for i in range(REPEAT):
    	c_baseacc, c_basediff, _, _ = utils.test(branchyNet, X, Y, main=True, batchsize=TEST_BATCHSIZE)
        res_basediff.append(c_basediff)
 
    print("LeNet accuracy is ", c_baseacc)
    print("LeNet time is ", sum(res_basediff)/REPEAT)
    print("\n")
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
    if cuda.available:
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
 
    acc, diff = measure_performance_branchynet(decoded_imgs.reshape(-1, 1, 28,28), Y_test_tmp, threshold=threshold)
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

def get_data():
    mnist = fetch_openml('mnist_784')
    x_all = mnist['data'].astype(np.float32) / 255
    y_all = mnist['target'].astype(np.int32)
    x_train, x_test = np.split(x_all, [60000])
    y_train, y_test = np.split(y_all, [60000])

    x_train = x_train.reshape([-1,1,28,28])
    x_test = x_test.reshape([-1,1,28,28])
    return x_train,y_train,x_test,y_test


_, _, Q_X_test, Q_Y_test = qmnist.get_qmnist("../datasets/qmnist/")
Q_Y_test = Q_Y_test.reshape(-1 , )
Q_data_shape = (Q_X_test.shape[0], 784)

# measure branchyNet
print("\nmeasure branchyNet")
measure_performance_branchynet(Q_X_test, Q_Y_test)
# measure BranchyNet with early exit
print("\n\nmeasure BranchyNet with early exit")
measure_performance_branchynet(Q_X_test, Q_Y_test, 2)
# measure LeNet
print("\n\nmeasure LeNet")
measure_performance_LeNet(Q_X_test, Q_Y_test)
# measure performance data,all data go into different exits
print("\n\nmeasure performance data,all data go into different exits")
measure_perf_and_time(Q_X_test, Q_Y_test, (-1, 784), 2)
