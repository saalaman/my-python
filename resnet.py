import os
import numpy as np
import tensorflow as tf
import h5py
import math
import tempfile
from data_set import *

hdf5_path ='E:/Graduation-Project/Event&NoEvent/HDF5'
model_save_path ='E:/Graduation-Project/test1/model/model_resnet'
# hdf5_path ='E:/Graduation-Project/data/HDF5/dataset64color.hdf5'
# model_save_path ='E:/Graduation-Project/test1/model/model_resnet'

class resnet_2classifier(object):

    def __init__(self, model_save_path = model_save_path):
        self.model_save_path = model_save_path
    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        """
        执行如图3中定义的标识块
        参数：
         X  - 输入形状张量（m，n_H_prev，n_W_prev，n_C_prev）
         kernel_size  - 整数，指定主路径的中间CONV窗口的形状
         filters  -  python的整数列表，定义主路径的CONV层中的过滤器数量
         stage  - 整数，用于命名图层，具体取决于它们在网络中的位置
         block  - 字符串/字符，用于命名图层，具体取决于它们在网络中的位置
         training - 训练或测试
        返回：
         X  - 标识块的输出，形状的张量（n_H，n_W，n_C）
        """
        # defining name basis
        block_name = 'res' + str(stage) + block  # res+整数+字符串
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input
            # first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
            #第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
            # 具有[batch, in_height, in_width, in_channels]这样的shape，
            # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，
            # 要求类型为float32和float64其中之一
            # X = tf.layers.batch_normalization(X, axis=1, training=training)
            X = tf.nn.relu(X)

            # second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)
            X = tf.nn.relu(X)

            # third
            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X,  training=training)

            # final step
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result

    def convolutional_block(self, X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
        """
    卷积块的实现如图4所定义
        参数：
         X  - 输入形状张量（m，n_H_prev，n_W_prev，n_C_prev）
         kernel_size  - 整数，指定主路径的中间CONV窗口的形状
         filters  -  python的整数列表，定义主路径的CONV层中的过滤器数量
         stage  - 整数，用于命名图层，具体取决于它们在网络中的位置
         block  - 字符串/字符，用于命名图层，具体取决于它们在网络中的位置
         train - 训练或测试
         stride  - 整数，指定要使用的步幅

        返回：
         X  - 卷积块的输出，形状的张量（n_H，n_W，n_C）
        """
        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            # first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X,  training=training)
            X = tf.nn.relu(X)

            # second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)
            X = tf.nn.relu(X)

            # third
            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, training=training)

            # shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            # final
            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result

    def deepnn(self, x_input, classes=9):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        Arguments:
        Returns:
        """
        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
        with tf.variable_scope('reference'):
            training = tf.placeholder(tf.bool, name='training')
            # stage 1
            w_conv1 = self.weight_variable([7, 7, 1, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')#得到64张33x33的feature map
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='VALID')
            # assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))#15x15x64

            # stage 2
            x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)#13x13x256
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)#11x11x256
            #X_input, kernel_size, in_filter, out_filters, stage, block, training
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            # stage 3
            x = self.convolutional_block(x, 3, 256, [128, 128, 512], 3, 'a', training)#9x9x512
            x = self.identity_block(x, 3, 512, [128, 128, 512], 3, 'b', training=training)#7x7x512
            x = self.identity_block(x, 3, 512, [128, 128, 512], 3, 'c', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], 3, 'd', training=training)

            # stage 4
            x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)#5x5x1024
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)#3x3x1024
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            # stage 5
            x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

            flatten = tf.layers.flatten(x)#一维化，加全连接层
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=classes, activation=tf.nn.softmax)

        return logits, keep_prob, training

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """weight_variable生成给定形状的权重变量."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable生成给定形状的偏差变量."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_op = tf.reduce_mean(correct_prediction)
        return accuracy_op

    def train(self, X_train, Y_train,learning_rates,step):
        features = tf.placeholder(tf.float32, [None, 224, 224, 1])
        labels = tf.placeholder(tf.int64, [None, 9])

        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None)

        logits, keep_prob, train_mode = self.deepnn(features)
        print('logits and labels',logits.shape, labels.shape)

        cross_entropy = self.cost(logits, labels)

        with tf.name_scope('adam_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(learning_rates).minimize(cross_entropy)

        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(step):
                X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
                train_step.run(
                    feed_dict={features: X_mini_batch, labels: Y_mini_batch, keep_prob: 0.8, train_mode: True})
                if i % 100 == 0:
                    train_cost = sess.run(cross_entropy, feed_dict={features: X_mini_batch,
                                                                    labels: Y_mini_batch, keep_prob: 1.0,
                                                                    train_mode: False})
                    print('step %d, training cost %g' % (i, train_cost))

            saver.save(sess, self.model_save_path)

    def evaluate(self, test_features, test_labels, name='test '):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 224, 224, 1])
        y_ = tf.placeholder(tf.int64, [None, 9])

        logits, keep_prob, train_mode = self.deepnn(x)
        accuracy = self.accuracy(logits, y_)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_path)
            accu = sess.run(accuracy, feed_dict={x: test_features, y_: test_labels, keep_prob: 1.0, train_mode: False})
            print('%s accuracy %g' % (name, accu))

def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    print(X.shape, Y.shape)
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(Y):
    one_hot_index = np.arange(len(Y)) * 2 + Y
    #print(one_hot_index.shape)
    #print('one_hot_index:{}'.format(one_hot_index))
    one_hot = np.zeros((len(Y), 2))
    one_hot.flat[one_hot_index] = 1
    #print('one_hot:{}'.format(one_hot))
    return one_hot

def process_orig_datasets(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig,X_val_orig,Y_val_orig ):
    """
    normalize x_train and convert y_train to one hot.
    :param datasets: X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes
    :return: X_train, Y_train, X_test, Y_test
    """
    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    X_val = X_val_orig / 255.

    # 将训练和测试标签转换为一个hot矩阵
    Y_train = convert_to_one_hot(Y_train_orig)
    Y_test = convert_to_one_hot(Y_test_orig)
    Y_val = convert_to_one_hot(Y_val_orig)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def main(_):
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # session = tf.Session(config=config, )
    data_dir = hdf5_path
    orig_data = load_dataset(data_dir)
    X_train, y_train, X_test, y_test,X_val,y_val = orig_data
    # y_test = (y_test.reshape(500,)).astype(int)
    # y_train = (y_train.reshape(3780,)).astype(int)
    # y_val = (y_val.reshape(500, )).astype(int)
    ########
    y_test = (y_test.reshape(150, )).astype(int)
    y_train = (y_train.reshape(1161, )).astype(int)
    y_val = (y_val.reshape(150, )).astype(int)
    ########
    X_train, y_train, X_test, y_test,X_val,y_val = process_orig_datasets(X_train, y_train, X_test, y_test,X_val,y_val )
    # print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    # X_train=X_train.reshape(3780,64,64,1)
    # X_test = X_test.reshape(500, 64, 64, 1)
    # X_val = X_val.reshape(500, 64, 64, 1)
    ###########
    X_train = X_train.reshape(1161, 224, 224, 1)
    X_test = X_test.reshape(150, 224, 224, 1)
    X_val = X_val.reshape(150, 224, 224, 1)
    ###########
    model = resnet_2classifier()
    learning_rates = 1e-4
    step = 200
    print('                  learning_rates',learning_rates,'             step',step)
    model.train(X_train, y_train,learning_rates,step)
    model.evaluate(X_train, y_train, '           training data')
    # model.evaluate(X_val, y_val,'              val data')
    model.evaluate(X_test, y_test, '              test data')

if __name__ == '__main__':
    tf.app.run(main=main)