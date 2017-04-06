from nntype import NNType
from neural_network import NeuralNetwork


class SimpleNeuron (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(1, 1)

        output = self.create_layer(self.X, 1, 1, 'W', 'b')

        self.set_hypothesis(output)
        self.set_cost_function(NNType.SQUARE_MEAN)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


gildong = SimpleNeuron()
x_data =[[1], [2], [3]]
y_data = [[1], [2], [3]]

gildong.learn(x_data, y_data, 2000, 20)
gildong.evaluate_linear(x_data, y_data)
gildong.test_linear([[7]])
gildong.show_error()

