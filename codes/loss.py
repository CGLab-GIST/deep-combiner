
import numpy as np
import tensorflow.compat.v1 as tf

def L1(y, _y):
    print('loss : L1')
    return tf.reduce_mean(tf.losses.absolute_difference(y, _y))


def L2(y, _y):
    print('loss : L2')
    return tf.reduce_mean(tf.squared_difference(y, _y))


def RELMSE(y, _y):
    eps = 1e-2
    print('loss : relMSE (eps %f)' % (eps))
    num = tf.square(tf.subtract(y, _y))
    denom = tf.reduce_mean(_y, axis=3, keepdims=True)
    relMse = num / (denom * denom + eps)
    relMseMean = tf.reduce_mean(relMse)
    return relMseMean


def minimizeAdamOptimizer(learning_rate, loss, name='adam'):
    return tf.train.AdamOptimizer(learning_rate, name=name).minimize(loss)