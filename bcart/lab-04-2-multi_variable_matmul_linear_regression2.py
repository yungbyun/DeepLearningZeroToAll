# Lab 4 Multi-variable linear regression
import tensorflow as tf

from regression import Regression
from mytype import MyType
from neural_network import NeuralNetwork


class MVLinearRegression (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(3, 1)

        output = self.create_layer(self.X, 3, 1, 'W', 'b')

        self.set_hypothesis(output)
        self.set_cost_function(MyType.LINEAR)
        self.set_optimizer(MyType.GRADIENTDESCENT, l_rate=1e-5)

    '''
    def my_log(self, x_data, y_data):
        cost_val, hy_val = self.sess.run([self.cost_function, self.hypothesis], feed_dict={self.X: x_data, self.Y: y_data})
        str = 'Error: {}, Answer: {}'.format(cost_val, hy_val)
        self.logs.append(str)
    '''
    '''
    [73.0, 80.0, 75.0]
    [93.0, 88.0, 93.0]
    [89.0, 91.0, 90.0]
    [96.0, 98.0, 100.0]
    [73.0, 66.0, 70.0]
    ->
    [ 154.35881042]
    [ 182.95147705]
    [ 181.85035706]
    [ 194.35533142]
    [ 142.036026]
    '''

x_dat = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_dat = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

gildong = MVLinearRegression()
gildong.learn(x_dat, y_dat, 2000, 40)
#gildong.print_log()
gildong.test_linear(x_dat)
gildong.show_error()


