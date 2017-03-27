from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plot
import numpy
import myplot


class TFVariable:
    import tensorflow as tf
    import numpy

    @staticmethod
    def get_var(name):
        return tf.Variable(numpy.random.randn(), name)

    @staticmethod
    def get_var_list(number):
        return tf.Variable(tf.random_uniform([number], -1.0, 1.0))  # 리스트로 리턴
