# Lab 4 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data

from neural_network import NeuralNetwork


class LinearRegressionFromFiles (NeuralNetwork):
    def init(self):
        self.set_placeholder(3, 1)
        self.set_weight_bias(3, 1)
        self.set_hypothesis(1)
        self.set_cost_function(1)
        self.set_optimizer(l_rate=1e-5)

    def my_log(self, x_data, y_data):
        pass

    '''
    [60, 70, 110]
    [90, 100, 80]
    ->
    [ 136.05517578]
    [ 189.76937866]
    '''

gildong = LinearRegressionFromFiles()
gildong.learn_batch(['data-01-test-score.csv', 'data-01-test-score.csv'], 2000, 200)
gildong.test([[60, 70, 110], [90, 100, 80]])

