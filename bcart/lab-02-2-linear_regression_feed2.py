# Lab 2 Linear Regression
import tensorflow as tf
from nntype import NNType
from neural_network import NeuralNetwork


class MVLogisticRegression (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(2, 1)

        output = self.create_layer(self.X, 2, 1, 'W', 'b')
        output = tf.sigmoid(output)

        self.set_hypothesis(output)
        self.set_cost_function(NNType.LOGISTIC)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


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
gildong.test_sigmoid([[6, 2]])

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
