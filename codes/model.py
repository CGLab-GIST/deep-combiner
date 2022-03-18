
import numpy as np
import tensorflow.compat.v1 as tf
from weighted_average import weighted_average

def kpn_conv2d(x, filter_size, out_channel, is_training=False, name='conv', relu=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        in_shape = x.get_shape().as_list()
        in_channel = in_shape[-1]
        
        filter_shape = [filter_size, filter_size, in_channel, out_channel]
        W = tf.get_variable(name='W', shape=filter_shape, dtype=tf.float32, initializer=tf.keras.initializers.glorot_normal(), trainable=True)
        b = tf.get_variable(name='b', shape=[out_channel], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=True)

        res = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
        res = tf.nn.bias_add(res, b ,name='bias')
        if relu:
            res = tf.nn.relu(res, name = 'relu')

        return res
		

def combination_kernel_single(rand, corr, net_out, name='recons'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sum_w = tf.reduce_sum(net_out, reduction_indices=-1, keepdims=True)
        wgt = tf.divide(net_out, tf.maximum(sum_w, 1e-10))
        res = weighted_average(rand - corr, wgt) + corr
        
        return res


def combination_kernel_multi(rand0, rand1, rand2, rand3, corr0, corr1, corr2, corr3, net_out, name='recons'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sum_w = tf.reduce_sum(net_out, reduction_indices=-1, keepdims=True)
        net_out = tf.divide(net_out, tf.maximum(sum_w, 1e-10))
        wgt0, wgt1, wgt2, wgt3 = tf.split(net_out, 4, axis=-1)

        wgtSum0 = tf.reduce_sum(wgt0, reduction_indices=-1, keepdims=True)
        wgtSum1 = tf.reduce_sum(wgt1, reduction_indices=-1, keepdims=True)
        wgtSum2 = tf.reduce_sum(wgt2, reduction_indices=-1, keepdims=True)
        wgtSum3 = tf.reduce_sum(wgt3, reduction_indices=-1, keepdims=True)

        res0 = weighted_average(rand0 - corr0, wgt0) + tf.multiply(corr0, wgtSum0)
        res1 = weighted_average(rand1 - corr1, wgt1) + tf.multiply(corr1, wgtSum1)
        res2 = weighted_average(rand2 - corr2, wgt2) + tf.multiply(corr2, wgtSum2)
        res3 = weighted_average(rand3 - corr3, wgt3) + tf.multiply(corr3, wgtSum3)
        res = res0 + res1 + res2 + res3

        return res


def KERNER_PREDICTING_NETWORK(x, _kernel_size, _buffer_size, is_training):
    conv1 = kpn_conv2d(x, 5, 100, is_training, name='convInput')
    hidden1 = kpn_conv2d(conv1, 5, 100, is_training, name='conv1')
    hidden2 = kpn_conv2d(hidden1, 5, 100, is_training, name='conv2')
    hidden3 = kpn_conv2d(hidden2, 5, 100, is_training, name='conv3')
    hidden4 = kpn_conv2d(hidden3, 5, 100, is_training, name='conv4')
    net_out = kpn_conv2d(hidden4, 5, _buffer_size * _kernel_size, is_training, name='conv5')

    return net_out


def COMBINER_SINGLE_BUFFER(x, _kernel_size, _h, _w, _b, is_training):
    corr, rand, normal, texture, depth = tf.split(x, [3, 3, 3, 3, 1], axis=-1)

    # depth normalization locally
    minDepth = tf.reduce_min(depth, reduction_indices=-1, keepdims=True)
    maxDepth = tf.reduce_max(depth, reduction_indices=-1, keepdims=True)
    depth = (depth - minDepth) / (maxDepth - minDepth + 0.001)

    feat = tf.concat([corr, rand, normal, texture, depth], axis=-1)
    net_out = KERNER_PREDICTING_NETWORK(feat, _kernel_size, 1, is_training)
    out_img = combination_kernel_single(rand, corr, net_out)

    return out_img


def COMBINER_MULTI_BUFFER(x, _kernel_size, _h, _w, _b, is_training):
    corr0, corr1, corr2, corr3, rand0, rand1, rand2, rand3, normal, texture, depth = tf.split(x, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], axis=-1)

    # depth normalization locally
    minDepth = tf.reduce_min(depth, reduction_indices=-1, keepdims=True)
    maxDepth = tf.reduce_max(depth, reduction_indices=-1, keepdims=True)
    depth = (depth - minDepth) / (maxDepth - minDepth + 0.001)

    feat = tf.concat([corr0, corr1, corr2, corr3, rand0, rand1, rand2, rand3, normal, texture, depth], axis=-1)
    net_out = KERNER_PREDICTING_NETWORK(feat, _kernel_size, 4, is_training)
    out_img = combination_kernel_multi(rand0, rand1, rand2, rand3, corr0, corr1, corr2, corr3, net_out)

    return out_img
