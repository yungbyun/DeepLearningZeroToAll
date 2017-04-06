# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from file2buffer import File2Buffer
import numpy as np
from nntype import NNType
from softmax_onehot import SoftmaxOnehot


class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(16, 1)
        #self.set_one_hot(7)

        logit = self.create_layer(self.X, 16, 7, 'W', 'b')
        output = tf.nn.softmax(logit)

        self.set_hypothesis(output)
        self.set_cost_function(NNType.SOFTMAX)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)

gildong = XXX()
gildong.learn_with_file('data-04-zoo.csv', 2000, 100)
gildong.evaluate_file_one_hot('data-04-zoo.csv', 7)
gildong.print_error()


'''
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
0.0778563
0.0724291
0.0677236
0.0636065
0.0599748
0.0567477
0.0538613
'''

