
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

# JH: custom_op from weighted_averaged_lib.so file
_module = tf.load_op_library('./weighted_average_lib.so')

@tf.RegisterGradient("WeightedAverage")
def _weighted_average_grad(op, grad):
    image = op.inputs[0]
    weights = op.inputs[1]
    grads = _module.weighted_average_grad(grad, image, weights)
    grads = tf.clip_by_value(grads, -1000000, 1000000)
    return [None, grads]

weighted_average = _module.weighted_average
