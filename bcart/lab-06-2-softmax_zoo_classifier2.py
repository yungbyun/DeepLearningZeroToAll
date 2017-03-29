# Lab 6 Softmax Classifier
from softmax_classifier import SoftMaxClassifier


class XXX (SoftMaxClassifier):
    def init(self):
        self.set_placeholder(16, 1)
        self.set_one_hot(7) #class number
        self.set_weight_bias(16, 7)
        self.set_hypothesis(3)
        self.set_cost_function(4)
        self.set_optimizer(0.1)

        '''
        ex) 0,0,1,0,0, 1,1,1,1,0, 0,1,0,1,0, 0,      3

        [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
        ->
        [  4.11042303e-04   7.76130008e-04   8.66002776e-03   9.85171258e-01
           1.73134496e-03   8.52992230e-07   3.24933371e-03]
        [3]

        100.00%
        '''

gildong = XXX()
x_data, y_data = gildong.load_file('data-04-zoo.csv')
#gildong.learn(x_data, y_data, 2000, 100)
gildong.learn_from_file('data-04-zoo.csv', 2000, 100)
gildong.test_argmax([[0,0,1,0,0, 1,1,1,1,0, 0,1,0,1,0, 0]])
gildong.recognition_rate(x_data, y_data)
#gildong.show_error()
