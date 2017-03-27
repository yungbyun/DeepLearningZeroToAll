# Lab 5 Logistic Regression Classifier
from neural_network import NeuralNetwork


class MVLogisticRegression (NeuralNetwork):
    def init(self):
        #self.init_for_multi_variable_logistic_regression()
        self.set_placeholder(2, 1)
        self.set_weight_bias(2, 1)
        self.set_hypothesis(2)
        self.set_cost_function(2)
        self.set_optimizer(l_rate=0.1)

    def my_log(self, x_data, y_data):
        pass

    '''
    [1, 2]
    [2, 3]
    [3, 1]
    [4, 3]
    [5, 3]
    [6, 2]
    ->
    [ 0.00138244]
    [ 0.0577572]
    [ 0.07596549]
    [ 0.92112178]
    [ 0.99383414]
    [ 0.99856013]
    '''

gildong = MVLogisticRegression()

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

gildong.learn(x_data, y_data, 5000, 200);
#gildong.show_error()
#gildong.print_weight()
gildong.test(x_data)
#gildong.recognition_rate(x_data, y_data)
