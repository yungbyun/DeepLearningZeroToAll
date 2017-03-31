# Lab 2 Linear Regression
import tensorflow as tf
from regression import Regression
from mytype import MyType
from neural_network import NeuralNetwork

class MVLogisticRegression (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(2, 1)
        W, b, output = self.create_layer(None, 2, 1, MyType.LOGISTIC)
        self.set_hypothesis(output)
        self.set_cost_function(MyType.LOGISTIC) #logistic
        self.set_optimizer(MyType.GRADIENTDESCENT, 0.1)


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
gildong.evaluate_sigmoid(x_data, y_data)
gildong.test_sigmoid([[6, 3]])

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
