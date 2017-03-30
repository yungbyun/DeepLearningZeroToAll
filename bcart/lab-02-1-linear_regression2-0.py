from regression import Regression
from mytype import MyType

class XXX (Regression):
    def init_network(self):
        self.set_placeholder(1, 1)
        self.set_weight_bias(1, 1)
        self.set_hypothesis(MyType.LINEAR)
        self.set_cost_function(MyType.LINEAR)
        self.set_optimizer(0.1)


gildong = XXX()
x_data =[[1], [2], [3]]
y_data = [[1], [2], [3]]

gildong.learn(x_data, y_data, 2000, 20)
gildong.test([[7]])
gildong.print_weight()
gildong.show_error()

