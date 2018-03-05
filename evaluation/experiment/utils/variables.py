import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer


def weight_variable(name, shape, regularization=None):
    regularizer = None
    if regularization is not None:
        regularizer = l2_regularizer(regularization)
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), regularizer=regularizer)


def bias_variable(name, shape, value=0.1, regularization=None):
    regularizer = None
    if regularization is not None:
        regularizer = l2_regularizer(regularization)
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value), regularizer=regularizer)
