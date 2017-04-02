# Lab 10 MNIST and Dropout
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from dropout_mnist_neural_network import DropoutMnistNeuralNetwork
from mytype import MyType


class XXX (DropoutMnistNeuralNetwork):

    def init_network(self):
        self.set_placeholder(784, 10)
        output_a = self.create_input_layer(784, 512, 'Wa')
        output_b = self.create_hidden_layer(output_a, 512, 512, 'Wb')
        output_c = self.create_hidden_layer(output_b, 512, 512, 'Wc')
        output_d = self.create_hidden_layer(output_c, 512, 512, 'Wd')
        hypo = self.create_output_layer(output_d, 512, 10, 'We')
        self.set_hypothesis(hypo)
        self.set_cost_function(MyType.SOFTMAX_LOGITS)
        self.set_optimizer(MyType.ADAM, 0.001)


gildong = XXX()
gildong.learn_mnist(15, 100)
gildong.evaluate()
gildong.classify_random()


'''
Epoch: 0001 cost = 0.447322626
Epoch: 0002 cost = 0.157285590
Epoch: 0003 cost = 0.121884535
Epoch: 0004 cost = 0.098128681
Epoch: 0005 cost = 0.082901778
Epoch: 0006 cost = 0.075337573
Epoch: 0007 cost = 0.069752543
Epoch: 0008 cost = 0.060884363
Epoch: 0009 cost = 0.055276413
Epoch: 0010 cost = 0.054631256
Epoch: 0011 cost = 0.049675195
Epoch: 0012 cost = 0.049125314
Epoch: 0013 cost = 0.047231930
Epoch: 0014 cost = 0.041290121
Epoch: 0015 cost = 0.043621063
Learning Finished!
Accuracy: 0.9804
'''
