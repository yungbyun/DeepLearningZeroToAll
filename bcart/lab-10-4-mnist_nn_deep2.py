# Lab 10 MNIST and Deep learning
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from mnist_neural_network import MnistNeuralNetwork
from mytype import MyType
from mnist_neural_network import MnistNeuralNetwork


'''
in:image (784) -> out:0~9 (10)
5 layers
Xavier initialization
RELU activation function
Hypothesis: Softmax, cf) One-hot encoding (argmax)
Cost function: Cross-Entropy, D(H,Y)
Optimizer: ADAM
'''
class XXX (MnistNeuralNetwork):
    def init_network(self):
        self.set_placeholder(784, 10)
        Wa, ba, output_a = self.create_layer(None, 784, 512, MyType.RELU, 'weight_a', 'bias_a')
        Wb, bb, output_b = self.create_layer(output_a, 512, 512, MyType.RELU, 'weight_b', 'bias_b')
        Wc, bc, output_c = self.create_layer(output_b, 512, 512, MyType.RELU, 'weight_c', 'bias_c')
        Wd, bd, output_d = self.create_layer(output_c, 512, 512, MyType.RELU, 'weight_d', 'bias_d')
        We, be, hypo = self.create_layer(output_d, 512, 10, MyType.LINEAR, 'weight_e', 'bias_e')
        self.set_hypothesis(hypo)
        self.set_cost_function(MyType.SOFTMAX_LOGITS)
        self.set_optimizer(MyType.ADAM, 0.001)

    def set_weight_initializer(self):
        self.xavier()


gildong = XXX()
gildong.learn_mnist(15, 100)
gildong.evaluate()
gildong.classify_random()
gildong.show_error()


'''
Epoch: 0001 cost = 0.329985125
Epoch: 0002 cost = 0.109935446
Epoch: 0003 cost = 0.073520600
Learning Finished!
Accuracy: 0.9724
Label:  [3]
Prediction:  [3]
'''
