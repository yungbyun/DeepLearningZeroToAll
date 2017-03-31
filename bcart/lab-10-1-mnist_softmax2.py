# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from softmax_del import Softmax
from mytype import MyType
from mnist_classifier_del import MnistClassifier
from mnist_neural_network import MnistNeuralNetwork

class XXX (MnistNeuralNetwork):
    def init_network(self):
        self.set_placeholder(784, 10)
        W, b, logits = self.create_layer(None, 784, 10, MyType.LINEAR)  # for logits
        self.set_hypothesis(logits)
        self.set_cost_function(MyType.SOFTMAX_LOGITS)
        self.set_optimizer(MyType.ADAM, 0.001)


gildong = XXX()
gildong.learn(15, 100)
gildong.classify_random_image()
gildong.evaluate()
gildong.show_error()


'''
Epoch: 0001 cost = 5.916487225
Epoch: 0002 cost = 1.868882108
Epoch: 0003 cost = 1.165184578
Epoch: 0004 cost = 0.895758619
Epoch: 0005 cost = 0.753617214
Epoch: 0006 cost = 0.665425729
Epoch: 0007 cost = 0.604189989
Epoch: 0008 cost = 0.559039789
Epoch: 0009 cost = 0.523200151
Epoch: 0010 cost = 0.494997439
Epoch: 0011 cost = 0.471757663
Epoch: 0012 cost = 0.451736393
Epoch: 0013 cost = 0.435549789
Epoch: 0014 cost = 0.421230042
Epoch: 0015 cost = 0.408108964
Learning Finished!
Accuracy: 0.9014
'''
