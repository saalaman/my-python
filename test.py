from __future__ import print_function
import keras
import numpy as np
import os
import h5py
import time
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

'''
第五步，测试测试的准确率
'''

hdf5_path ='E:/Graduation-Project/Event&NoEvent/HDF5/dataset506_all.hdf5'
hdf5_path_train ='E:/Graduation-Project/Event&NoEvent/HDF5/dataset112_val.hdf5'
model_save_path ='E:/Graduation-Project/test1/model_keras/model_resnet425/cifar10_ResNet14v1_model.051.h5'
# no_even = 'E:/Graduation-Project/Event&NoEvent/TEST_No_Even'
# hdf5_path = 'E:/homework_cilent/data_set/HDF5/dataset96color.hdf5'
# path = 'E:/Graduation-Project/Event&NoEvent/TEST_15s'

num_classes = 2
np.set_printoptions(threshold = 1e6)
pd.set_option('display.width', None)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
# #############测试准确率，用的有标签的H5文件###########################################
# # Load the data.
# def load_dataset(path1,path2):
#     dataset = h5py.File(path1, "r")
#     dataset2 = h5py.File(path2, "r")
#
#     test_set_x_orig = np.array(dataset["X_test"][:])  # your test set features
#     test_set_y_orig = np.array(dataset["y_test"][:])  # your test set labels
#
#     train_set_x_orig = np.array(dataset2["X_train"][:])  # your train set features
#     train_set_y_orig = np.array(dataset2["y_train"][:])  # your train set labels
#
#     #classes = np.array(test_dataset["list_classes"][:])  # the list of classes
#     return test_set_x_orig, train_set_x_orig , test_set_y_orig ,train_set_y_orig
#
# orig_data = load_dataset(hdf5_path,hdf5_path_train)
# x_test,  x_train, y_test, y_train = orig_data
#
# # Normalize data.
# x_test = x_test.astype('float32') / 255
# # x_test = np.expand_dims(x_test, axis=0)
# x_train = x_train.astype('float32') / 255
#
# x_train_mean = np.mean(x_train, axis=0)
# x_train -= x_train_mean
# x_test -= x_train_mean
#
# # Convert class vectors to binary class matrices.
# y_test = keras.utils.to_categorical(y_test, num_classes)
# # print(y_test)
#
#
# model = load_model(model_save_path)
# #模型评估
# scores_test = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores_test[0])
# print('Test accuracy:', scores_test[1])
# #########################################################################
##############仅为得到顺序为时间序列的标签预测结果##########################
# Load the data.
def load_dataset(path1,path2):
    dataset = h5py.File(path1, "r")
    dataset2 = h5py.File(path2, "r")

    test_set_x_orig = np.array(dataset["X_test"][:])  # your test set features
    # test_set_y_orig = np.array(dataset["y_test"][:])  # your test set labels
    train_set_x_orig = np.array(dataset2["X_train"][:])  # your train set features
    # train_set_y_orig = np.array(dataset2["y_train"][:])  # your train set labels

    #classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    return test_set_x_orig, train_set_x_orig

orig_data = load_dataset(hdf5_path,hdf5_path_train)
x_test,  x_train = orig_data

# Normalize data.
x_test = x_test.astype('float32') / 255
# x_test = np.expand_dims(x_test, axis=0)
x_train = x_train.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean


model = load_model(model_save_path)
#########################################################################
pre = model.predict(x_test)
label_pre = np.argmax(pre, axis=1)
# label_pre = label_pre.reshape(8,283)
# print(pre)
print(label_pre)
# for i in range(811):
#     if label_pre[i] == 1:
#         print(i)
# 格式化成2016-03-20 11:45:39形式
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(model_save_path)

np.savetxt("E:/Graduation-Project/test1/data_nor.txt",label_pre) #缺省按照'%.18e'格式保存数据，以空格分隔
print(np.loadtxt("E:/Graduation-Project/test1/data_nor.txt"))











