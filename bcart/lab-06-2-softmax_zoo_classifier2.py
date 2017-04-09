# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from nntype import NNType
from neural_network_one_hot import NeuralNetworkOneHot


class XXX (NeuralNetworkOneHot):
    def init_network(self):
        self.set_placeholder(16, 1)

        self.target_to_one_hot(7)

        logits = self.create_layer(self.X, 16, 7, 'W', 'b')
        hypothesis = self.softmax(logits)

        self.set_hypothesis(hypothesis)
        self.set_cost_function_with_one_hot(logits, self.get_one_hot()) #not hypothesis, but logits
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


gildong = XXX()
xdata, ydata = gildong.load_file('data-04-zoo.csv')
gildong.learn(xdata, ydata, 2000, 100)
gildong.print_error()
gildong.evaluate('data-04-zoo.csv')
gildong.show_error()


'''
# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
'''

'''
Step:     0	Loss: 5.10635
Step:   100	Loss: 0.80030
Step:   200	Loss: 0.48635
Step:   300	Loss: 0.34942
Step:   400	Loss: 0.27165
Step:   500	Loss: 0.22188
Step:   600	Loss: 0.18692
Step:   700	Loss: 0.16078
Step:   800	Loss: 0.14046
Step:   900	Loss: 0.12429
Step:  1000	Loss: 0.11121
Step:  1100	Loss: 0.10050
Step:  1200	Loss: 0.09163
Step:  1300	Loss: 0.08418
Step:  1400	Loss: 0.07786
Step:  1500	Loss: 0.07243
Step:  1600	Loss: 0.06772
Step:  1700	Loss: 0.06361
Step:  1800	Loss: 0.05997
Step:  1900	Loss: 0.05675
Step:  2000	Loss: 0.05386
Acc: 100.00%
'''