# coding=utf-8
__author__ = 'moonkey'

#from keras import models, layers
import logging
import numpy as np
# from src.data_util.synth_prepare import SynthGen

#import keras.backend as K
import tensorflow as tf

BN_EPSILON = 0.001

def max_3x2pool(incoming, name):
    '''
    max pooling on 2 dims.
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 3, 2, 1), strides=(1, 3, 2, 1), padding='VALID')

def max_2x2pool(incoming, name):
    '''
    max pooling on 2 dims.
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')


def max_2x1pool(incoming, name):
    '''
    max pooling only on image width
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), padding='VALID')



def var_random(name, shape, regularizable=True):
    '''
    Initialize a random variable using xavier initialization.
    Add regularization if regularizable=True
    :param name:
    :param shape:
    :param regularizable:
    :return:
    '''
    v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizable:
        with tf.name_scope(name + '/Regularizer/'):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v))
    return v


def batch_norm(incoming, is_training):
    '''
    batch normalization
    :param incoming:
    :param is_training:
    :return:
    '''
    return tf.contrib.layers.batch_norm(incoming, is_training=is_training, scale=True, decay=0.99)

def conv_bn_relu_layer(input_layer, filter_shape, stride,name):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    filter = var_random(name=name, shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_norm(conv_layer, True)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride,name):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    bn_layer = batch_norm(input_layer, True)
    relu_layer = tf.nn.relu(bn_layer)

    filter =var_random(name=name, shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, name,first_block=False,):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 1
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_'+name):
        if first_block:
            filter = var_random(name='conv1_in_'+name, shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride,name='conv1_in_'+name)

    with tf.variable_scope('conv2_in_'+name):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1,name='conv2_in_'+name)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        padded_input = tf.pad(input_layer, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def dropout(incoming, is_training, keep_prob=0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)


class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()

    """

    def __init__(self, input_tensor, is_training):
        self._build_network(input_tensor, is_training)

    def _build_network(self, input_tensor, is_training):
        """
        https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
        :return:
        """
        print('input_tensor dim: {}'.format(input_tensor.get_shape()))
        net = tf.transpose(input_tensor, perm=[0, 2, 3, 1]) # 输入的tensor shape为(batch,channel,h,w)
        net = tf.add(net, (-128.0))
        net = tf.multiply(net, (1/128.0)) # normalize 归一化

        net = conv_bn_relu_layer(net, [3 , 3, 3, 64], stride=1,name='conv_out_1')

        net = residual_block(net, 128,first_block=True,name='block_1')
        

        # print(net.get_shape().as_list())
        net = bn_relu_conv_layer(net,[3,3,128,128],stride=1,name='conv_out_2')
        net = max_3x2pool(net,'pooling_1')

        net = residual_block(net, 256, name='block_2')
        net = max_2x2pool(net, 'pooling_2') # 高度减半 宽度不变

        net = residual_block(net, 512, name='block_3')
        net=  max_2x1pool(net,'pooling_3')

        net = residual_block(net, 512, name='block_4')
        net = max_2x1pool(net, 'pooling_4')

        net=bn_relu_conv_layer(net,[3,3,512,512],stride=1,name='conv_out_3')
        net = max_2x1pool(net, 'pooling_5')

        net = dropout(net, is_training)

        print('CNN outdim before squeeze: {}'.format(net.get_shape()))  # 1x32x100 -> 24x512

        net = tf.squeeze(net,axis=1)

        print('CNN outdim: {}'.format(net.get_shape()))
        self.model = net

    def tf_output(self):
        # if self.input_tensor is not None:
        return self.model
    '''
    def __call__(self, input_tensor):
        return self.model(input_tensor)
    '''
    def save(self):
        pass



