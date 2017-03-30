# Lab 7 Learning rate and Evaluation
from mnist_classifier import MnistClassifier
from mytype import MyType


class XXX (MnistClassifier):
    def init_network (self):
        self.set_placeholder(784, 10) #28 * 28 = 784, 0~9 digits -> num_of_input, num_of_neuron
        self.set_weight_bias(784, 10)
        self.set_hypothesis(MyType.SOFTMAX)
        self.set_cost_function(MyType.SOFTMAX)
        self.set_optimizer(MyType.GRADIENTDESCENT, 0.1)

    def epoch_process(self, avg_err, x_data, y_data):
        str = 'Error in epoch = {:.9f}'.format(avg_err)
        self.logs.append(str)


gildong = XXX()
gildong.learn(15, 100)
gildong.classify_random_image()
gildong.evaluate()
#gildong.show_errors()
gildong.print_log()
gildong.show_errors()



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

