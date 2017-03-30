# Lab 9 XOR
# This example does not work
import numpy as np
from mytype import MyType
from neural_network import NeuralNetwork


class XXX (NeuralNetwork) :
    def init_network(self):
        self.set_placeholder(2, 1)
        Wa, ba, output1 = self.create_layer(None, 2, 2, 'weight_a', 'bias_a', MyType.LOGISTIC)
        Wb, bb, hypo = self.create_layer(output1, 2, 1, 'weight_b', 'bias_b', MyType.LOGISTIC)
        self.set_hypothesis(hypo)
        self.set_cost_function(MyType.LOGISTIC)
        self.set_optimizer(0.1)

    def my_log(self, i, xdata, ydata):
        pass


gildong  = XXX()
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ydata = np.array([[0], [1], [1], [0]], dtype=np.float32)
gildong.learn(xdata, ydata, 10000, 100)
gildong.test([[0, 1], [1, 0]])
gildong.evaluate(xdata, ydata)
gildong.show_error()


'''
when the loop is 3001,
Hypothesis:  [[ 0.18309775]
 [ 0.72479963]
 [ 0.8890698 ]
 [ 0.13911283]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''
