# Lab 9 XOR
# This example does not work
import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from mytype import MyType

class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(2, 1)
        Wa, ba, output1 = self.create_layer(None, 2, 10, 'weight_a', 'bias_a', MyType.LOGISTIC)  # input
        Wb, bb, output2 = self.create_layer(output1, 10, 10, 'weight_a', 'bias_a', MyType.LOGISTIC)  # hidden1
        Wc, bc, output3 = self.create_layer(output2, 10, 10, 'weight_a', 'bias_a', MyType.LOGISTIC)  # hidden2
        Wd, bd, hypothesis = self.create_layer(output3, 10, 1, 'weight_a', 'bias_a', MyType.LOGISTIC)  # output
        self.set_hypothesis(hypothesis)
        self.set_cost_function(MyType.LOGISTIC)
        self.set_optimizer(0.1)


gildong = XXX()
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ydata = np.array([[0], [1], [1], [0]], dtype=np.float32)
gildong.learn(xdata, ydata, 10000, 100)
#gildong.show_error()
gildong.evaluate(xdata, ydata)

'''
Hypothesis:  [[  7.80512521e-04]
 [  9.99238133e-01]
 [  9.98379230e-01]
 [  1.55658729e-03]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0

'''
