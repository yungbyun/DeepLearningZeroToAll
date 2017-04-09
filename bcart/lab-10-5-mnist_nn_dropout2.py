# Lab 10 MNIST and Dropout
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from dropout_mnist_neural_network import DropoutMnistNeuralNetwork
from nntype import NNType


class XXX (DropoutMnistNeuralNetwork):

    def init_network(self):
        self.set_placeholder(784, 10)

        # 1
        L1 = self.create_layer(self.X, 784, 512, 'Wa', 'ba')
        L1 = tf.nn.relu(L1)  # X
        L1 = tf.nn.dropout(L1, keep_prob = self.DO)

        # 2
        L2 = self.fully_connected_layer(L1, 512, 512, 'Wb')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.dropout(L2, keep_prob = self.DO)

        # 3
        L3 = self.fully_connected_layer(L2, 512, 512, 'Wc')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.dropout(L3, keep_prob = self.DO)

        # 4
        L4 = self.fully_connected_layer(L3, 512, 512, 'Wd')
        L4 = tf.nn.relu(L4)
        L4 = tf.nn.dropout(L4, keep_prob = self.DO)

        # 5
        hypo = self.fully_connected_layer(L4, 512, 10, 'We')
        self.set_hypothesis(hypo)

        self.set_cost_function(NNType.SOFTMAX_LOGITS)
        self.set_optimizer(NNType.ADAM, 0.001)

    def log_for_epoch(self, i, xdata, ydata):
        print('Cost:', self.sess.run(self.cost_function, feed_dict={self.X: xdata, self.Y: ydata, self.DO: 0.7}))


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


Start learning:
.
Done!

Recognition rate : 0.9514
Label [8]
Classified [8]
'''
