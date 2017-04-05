# Lab 7 Learning rate and Evaluation
from mnist_classifier_del import MnistClassifier
from mytype import MyType
from mnist_neural_network import MnistNeuralNetwork
import tensorflow as  tf


class XXX (MnistNeuralNetwork):
    def init_network (self):
        self.set_placeholder(784, 10) #28 * 28 = 784, 0~9 digits -> num_of_input, num_of_neuron

        L = self.create_layer(self.X, 784, 10, MyType.SOFTMAX, 'Wa', 'ba')
        L = tf.nn.softmax(L)

        self.set_hypothesis(L)

        self.set_cost_function(MyType.SOFTMAX)
        self.set_optimizer(MyType.GRADIENTDESCENT, 0.1)

    def epoch_process(self, avg_err, x_data, y_data):
        str = 'Error in epoch = {:.9f}'.format(avg_err)
        self.logs.append(str)


gildong = XXX()
gildong.learn_mnist(15, 100)
gildong.evaluate()
gildong.classify_random()
gildong.print_log()
gildong.show_error()

'''
Recognition rate : 0.896
Error in epoch = 2.827615572
Error in epoch = 1.061562471
Error in epoch = 0.837592054
Error in epoch = 0.733635712
Error in epoch = 0.670395139
Error in epoch = 0.625074675
Error in epoch = 0.590520532
Error in epoch = 0.564253636
Error in epoch = 0.541424411
Error in epoch = 0.522518319
Error in epoch = 0.506355120
Error in epoch = 0.492517450
Error in epoch = 0.479899605
Error in epoch = 0.469007378
Error in epoch = 0.459114127
'''

'''
Error in epoch = 2.827615563
Error in epoch = 1.062452680
Error in epoch = 0.838201799
...
Ended!
Label [4]
Classified [4]
Recognition rate : 0.8976
'''

