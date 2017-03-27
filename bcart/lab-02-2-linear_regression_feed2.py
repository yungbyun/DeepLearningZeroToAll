# Lab 2 Linear Regression
import tensorflow as tf
from neural_network import NeuralNetwork


class MVLogisticRegression (NeuralNetwork):
    def init(self):
        self.set_placeholder(2, 1)
        self.set_weight_bias(2, 1)
        self.set_hypothesis(2) #logistic
        self.set_cost_function(2) #logistic
        self.set_optimizer(0.1)

    def my_log(self, x_data, y_data):
        pass

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

gildong = MVLogisticRegression()
gildong.learn(x_data, y_data, 4000, 100)
gildong.test(x_data)

'''
[1, 2]
[2, 3]
[3, 1]
[4, 3]
[5, 3]
[6, 2]
->
[ 0.00224411]
[ 0.06879371]
[ 0.09333074]
[ 0.90649462]
[ 0.99107587]
[ 0.9977513]
'''
