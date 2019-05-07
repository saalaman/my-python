from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import os
import h5py
import time
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dropout
'''
第三步，用训练集训练网络
'''

test_path = 'E:/Graduation-Project/Event&NoEvent/HDF5/dataset430_nor.hdf5'#测试集数据做验证集
hdf5_path ='E:/Graduation-Project/Event&NoEvent/HDF5/dataset112_val.hdf5'#训练集数据
model_save_path ='E:/Graduation-Project/test1/model_keras/model_506'#模型保存地址
# hdf5_path = 'E:/homework_cilent/data_set/HDF5/dataset96color.hdf5'
# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 100
data_augmentation = True
num_classes = 2

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 4
# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 2


# Load the CIFAR10 data.

def load_dataset(path,path2):
    dataset = h5py.File(path, "r")
    testset = h5py.File(path2, "r")

    train_set_x_orig = np.array(dataset["X_train"][:])  # your train set features
    train_set_y_orig = np.array(dataset["y_train"][:])  # your train set labels

    val_set_x_orig = np.array(testset["X_test"][:])  # your test set features
    val_set_y_orig = np.array(testset["y_test"][:])  # your test set labels

    test_set_x_orig = np.array(dataset["X_test"][:])  # your test set features
    test_set_y_orig = np.array(dataset["y_test"][:])  # your test set labels

    # test_set_x_orig = np.array(dataset["X_test"][:])  # your test set features
    # test_set_y_orig = np.array(dataset["y_test"][:])  # your test set labels

    #classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    return train_set_x_orig, train_set_y_orig,test_set_x_orig,test_set_y_orig, val_set_x_orig, val_set_y_orig

orig_data = load_dataset(hdf5_path,test_path)
x_train, y_train, x_test, y_test ,x_val, y_val = orig_data

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_val -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
# print(x_val.shape[0], 'val samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

def lr_schedule(epoch):
    lr = 5e-4
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    conv = Conv2D(num_filters,
                  #filters：卷积核的数目（即输出的维度）
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  #补0策略，为“valid”, “same”
                  # “valid”代表只进行有效的卷积，即对边界数据不处理。
                  # “same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同，
                  # 因为卷积核移动时在边缘会出现大小不够的情况。
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-2))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x#bn+relu+conv

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:#例如ResNet20
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)#ResNet20:3

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):#RELU{[(conv+bn+relu)+(conv+bn)]+x(shortcut)}
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)#conv+bn+relu
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)#conv+bn
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)#shortcut
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Dropout(0.3, noise_shape=None, seed=None)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Dropout(0.3, noise_shape=None, seed=None)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

np.set_printoptions(threshold = 1e6)
pd.set_option('display.width', None)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth, num_classes = num_classes)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes = num_classes)
#训练与评估
#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
#打印模型
model.summary()

print(model_type)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# kf = KFold(n_splits=10,shuffle=True)
# i = 0
# for train_index , test_index in kf.split(x_train,y_train):
#     i = i+1
#     # Prepare model model saving directory.
#     save_dir = model_save_path
#     model_name =str(i)+ 'cifar10_%s_model.{epoch:03d}.h5' % model_type
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     filepath = os.path.join(save_dir, model_name)
#
#     # Prepare callbacks for model saving and for learning rate adjustment.
#     checkpoint = ModelCheckpoint(filepath=filepath,
#                                  monitor='val_acc',
#                                  verbose=1,
#                                  save_best_only=True)
#
#     lr_scheduler = LearningRateScheduler(lr_schedule)
#
#     lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                    cooldown=0,
#                                    patience=5,
#                                    min_lr=0.5e-6)
#
#     callbacks = [checkpoint, lr_reducer, lr_scheduler]
#
#     # Run training, with or without data augmentation.
#     if not data_augmentation:
#         print('Not using data augmentation.')
#         model.fit(x_train, y_train,
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   validation_data=(x_train[test_index], y_train[test_index]),
#                   shuffle=True,
#                   callbacks=[callbacks])
#     else:
#         print('Using real-time data augmentation.')
#         # This will do preprocessing and realtime data augmentation:
#         datagen = ImageDataGenerator(
#             # set input mean to 0 over the dataset
#             featurewise_center=False,
#             # set each sample mean to 0
#             samplewise_center=False,
#             # divide inputs by std of dataset
#             featurewise_std_normalization=False,
#             # divide each input by its std
#             samplewise_std_normalization=False,
#             # apply ZCA whitening
#             zca_whitening=False,
#             # epsilon for ZCA whitening
#             zca_epsilon=1e-06,
#             # randomly rotate images in the range (deg 0 to 180)
#             rotation_range=0,
#             # randomly shift images horizontally
#             width_shift_range=0.1,
#             # randomly shift images vertically
#             height_shift_range=0.1,
#             # set range for random shear
#             shear_range=0.,
#             # set range for random zoom
#             zoom_range=0.,
#             # set range for random channel shifts
#             channel_shift_range=0.,
#             # set mode for filling points outside the input boundaries
#             fill_mode='nearest',
#             # value used for fill_mode = "constant"
#             cval=0.,
#             # randomly flip images
#             horizontal_flip=True,
#             # randomly flip images
#             vertical_flip=False,
#             # set rescaling factor (applied before any other transformation)
#             rescale=None,
#             # set function that will be applied on each input
#             preprocessing_function=None,
#             # image data format, either "channels_first" or "channels_last"
#             data_format=None,
#             # fraction of images reserved for validation (strictly between 0 and 1)
#             validation_split=0.0)
#
#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train)
#
#     # Fit the model on the batches generated by datagen.flow().
#     train_log = model.fit_generator(datagen.flow(x_train[train_index], y_train[train_index], batch_size=batch_size),
#                         validation_data=(x_train[test_index], y_train[test_index]),
#                         epochs=epochs, verbose=1, workers=4,
#                         callbacks=callbacks)

    #模型评估
# Prepare model model saving directory.
save_dir = model_save_path
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    train_log = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_val, y_val),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', scores[0])
print('Train accuracy:', scores[1])

scores_test = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores_test[0])
print('Test accuracy:', scores_test[1])

scores_val = model.evaluate(x_val, y_val, verbose=1)
print('Val loss:', scores_val[0])
print('Val accuracy:', scores_val[1])

pre = model.predict(x_val)
label_pre = np.argmax(pre, axis=1)
# label_pre = label_pre.reshape(7,323)
print(label_pre)

# pre = model.predict(x_test,verbose=1)
# label_pre = np.argmax(pre, axis=1)
# print(label_pre)
# 格式化成2016-03-20 11:45:39形式
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# plot the training loss and accuracy
plt.style.use("ggplot")

# plt.figure(1)
# plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
# plt.title("Training Loss")
# plt.show()
#
# plt.figure(2)
# plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
# plt.title("Val Loss")
# plt.show()
#
# plt.figure(3)
# plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
# plt.title("train_acc")
# plt.show()
#
# plt.figure(4)
# plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
# plt.title("val_acc")
# plt.show()

# plt.title("Training Loss and Accuracy on sar classifier")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper right")
# plt.savefig("Loss_Accuracy_alexnet_{:d}e.jpg".format(epochs))
# plt.show()

