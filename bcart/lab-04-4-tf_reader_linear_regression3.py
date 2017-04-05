# Lab 4 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data
from mytype import MyType
from neural_network import NeuralNetwork


class LinearRegressionFromFiles (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(3, 1)

        output = self.create_layer(self.X, 3, 1, MyType.LINEAR, 'W', 'b')

        self.set_hypothesis(output)
        self.set_cost_function(MyType.LINEAR)
        self.set_optimizer(MyType.GRADIENTDESCENT, l_rate=1e-5)

    def my_log(self, i, x_data, y_data):
        pass

    '''
    [60, 70, 110]
    [90, 100, 80]
    ->
    [ 136.05517578]
    [ 189.76937866]
    '''

gildong = LinearRegressionFromFiles()
gildong.learn_with_files(['data-01-test-score.csv', 'data-01-test-score.csv'], 2000, 200)
gildong.test_linear([[60, 70, 110], [90, 100, 80]])

