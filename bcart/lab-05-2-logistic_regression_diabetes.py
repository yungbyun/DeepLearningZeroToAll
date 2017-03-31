# Lab 5 Logistic Regression Classifier
from neural_network import NeuralNetwork
from mytype import MyType


class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(8, 1)
        W, b, output = self.create_layer(None, 8, 1, MyType.LOGISTIC)
        self.set_hypothesis(output)
        self.set_cost_function(MyType.LOGISTIC)
        self.set_optimizer(MyType.GRADIENTDESCENT, l_rate=0.01)

    def my_log(self, i, x_data, y_data):
        pass

        '''
        1200 0.654603
        1400 0.640737
        1600 0.62813
        1800 0.616668
        2000 0.606246
        [[ 0.6939525]]

         [ 0.55056906]
         [ 0.71810943]
         [ 0.72589421]
         [ 0.58412576]
         [ 0.73007631]]

         [ 1.]
         [ 1.]]
        Accuracy:  0.642951


        [0.176471, 0.155779, 0, 0, 0, 0.052161, -0.952178, -0.733333]
        ->
        [ 0.6939525]

        '''

gildong = XXX()
gildong.learn_with_file('data-03-diabetes.csv', 2000, 200) #10000, 200
gildong.test_sigmoid([[0.176471,0.155779,0,0,0,0.052161,-0.952178,-0.733333]])
from file2buffer import File2Buffer
f2b = File2Buffer()
f2b.file_load('data-03-diabetes.csv')
gildong.evaluate_sigmoid(f2b.x_data, f2b.y_data)
gildong.show_error()
