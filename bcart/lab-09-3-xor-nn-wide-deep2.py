# Lab 9 XOR
# This example does not work
import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from mytype import MyType

class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(2, 1)

        L1 = self.create_layer(self.X, 2, 10, MyType.LOGISTIC, 'weight_a', 'bias_a')  # input
        L1 = tf.sigmoid(L1)

        L2 = self.create_layer(L1, 10, 10, MyType.LOGISTIC, 'weight_a', 'bias_a')  # hidden1
        L2 = tf.sigmoid(L2)

        L3 = self.create_layer(L2, 10, 10, MyType.LOGISTIC, 'weight_a', 'bias_a')  # hidden2
        L3 = tf.sigmoid(L3)

        L4 = self.create_layer(L3, 10, 1, MyType.LOGISTIC, 'weight_a', 'bias_a')  # output
        L4 = tf.sigmoid(L4)

        self.set_hypothesis(L4)

        self.set_cost_function(MyType.LOGISTIC)
        self.set_optimizer(MyType.GRADIENTDESCENT, 0.1)


gildong = XXX()
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ydata = np.array([[0], [1], [1], [0]], dtype=np.float32)
gildong.learn(xdata, ydata, 10000, 100)
gildong.evaluate_sigmoid(xdata, ydata)
gildong.show_error()


'''
Hypothesis:  [[  7.80512521e-04]
 [  9.99238133e-01]
 [  9.98379230e-01]
 [  1.55658729e-03]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  100.0

'''
