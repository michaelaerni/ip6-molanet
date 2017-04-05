import tensorflow as tf


def leaky_relu(features, alpha=0.0):
# TODO: Docstring
    return tf.maximum(alpha * features, features)
