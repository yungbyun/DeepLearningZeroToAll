# Lab 4 Multi-variable linear regression

from mytype import MyType
from neural_network import NeuralNetwork


class LinearRegressionFromFile (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(3, 1)

        output = self.create_layer(self.X, 3, 1, MyType.LINEAR, 'W', 'b')

        self.set_hypothesis(output)
        self.set_cost_function(MyType.LINEAR)
        self.set_optimizer(MyType.GRADIENTDESCENT, l_rate=1e-5)

    def my_log(self, i, x_data, y_data):
        pass

    '''
    [100, 70, 101]
    ->
    [ 181.73277283]


    [60, 70, 110]
    [90, 100, 80]
    ->
    [ 145.86265564]
    [ 187.23130798]
    '''

gildong = LinearRegressionFromFile()
gildong.learn_with_file('data-01-test-score.csv', 2000, 10)
gildong.test_linear([[100, 70, 101]])
gildong.test_linear([[60, 70, 110], [90, 100, 80]])

