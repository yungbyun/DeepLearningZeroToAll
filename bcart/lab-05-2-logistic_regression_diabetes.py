# Lab 5 Logistic Regression Classifier
from regression import Regression
from mytype import MyType

class XXX (Regression):
    def init_network(self):
        self.set_placeholder(8, 1)
        self.set_weight_bias(8, 1)
        self.set_hypothesis(MyType.LOGISTIC)
        self.set_cost_function(MyType.LOGISTIC)
        self.set_optimizer(l_rate=0.01)

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
gildong.learn_from_file('data-03-diabetes.csv', 2000, 200) #10000, 200
gildong.test([[0.176471,0.155779,0,0,0,0.052161,-0.952178,-0.733333]])
#gildong.print_weight()
