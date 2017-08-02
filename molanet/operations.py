from typing import Callable

import tensorflow as tf


# TODO: Docstrings


def leaky_relu(features, alpha=0.0):
    return tf.maximum(alpha * features, features)


def select_device(use_gpu: bool) -> Callable[[tf.Operation], str]:
    def _selector(op: tf.Operation) -> str:
        # Do not assign device placement for variables as it breaks Tensorflow somehow
        if op.type == "VariableV2":
            return ""

        return "/gpu:0" if use_gpu else "/cpu:0"
    return _selector


def use_cpu():
    return tf.device("/cpu:0")
