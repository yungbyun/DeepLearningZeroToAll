# Lab 4 Multi-variable linear regression
from bcart.backup.linear_regression import LinearRegression


class LinearRegressionFromFile (LinearRegression):
    def init(self):
        #x_col = len(x_data[0])
        #y_col = len(y_data[0])
        #print(x_col, y_col) # 3, 1

        self.set_placeholder(3, 1)
        self.set_weight_bias(3, 1)
        self.set_hypothesis(1)
        self.set_cost_function(1)
        self.set_optimizer(l_rate=1e-5)

    def my_log(self, x_data, y_data):
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
gildong.learn_from_file('data-01-test-score.csv', 2000, 10)
gildong.test([[100, 70, 101]])
#gildong.test([[60, 70, 110], [90, 100, 80]])


