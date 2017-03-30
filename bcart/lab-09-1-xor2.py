# Lab 9 XOR
# This example does not work
import tensorflow as tf
import numpy as np
from regression import Regression
from mytype import MyType


class XXX (Regression) :
    def init_network(self):
        self.set_placeholder(2, 1)
        self.set_weight_bias(2, 1)
        self.set_hypothesis(MyType.LOGISTIC)
        self.set_cost_function(MyType.LOGISTIC)
        self.set_optimizer(0.1)

    def my_log(self, i, x_data, y_data):
        super().my_log(i, x_data, y_data)


gildong = XXX()
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
gildong.learn(x_data, y_data, 4000, 100)
gildong.evaluate(x_data, y_data)
gildong.show_error()
#gildong.print_log()


'''
4000 0.693147

Hypothesis:  [[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
Correct:  [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
Accuracy:  0.5
'''
