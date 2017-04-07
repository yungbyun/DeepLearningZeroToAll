# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
from neural_network import NeuralNetwork
from nntype import NNType


class XXX (NeuralNetwork):
    Y = None
    Y_one_hot_reshaped = None

    def set_placeholder(self, num_of_input, num_of_output):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.int32, [None, num_of_output])  # 0 ~ 6

    def set_cost_function_with_one_hot(self, logits, reshaped):
        # Cross entropy cost/loss
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=reshaped)
        cost = tf.reduce_mean(cost_i)
        self.cost_function = cost

    def evaluate(self):
        x_data, y_data = self.load_file('data-04-zoo.csv')

        prediction = tf.argmax(self.hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(self.Y_one_hot_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = self.sess.run(accuracy, feed_dict={self.X: x_data, self.Y: y_data})
        print("Acc: {:.7%}".format(acc))

    def run(self):
        tf.set_random_seed(777)  # for reproducibility

        x_data, y_data = self.load_file('data-04-zoo.csv')
        print("here", x_data.shape, y_data.shape)

        nb_classes = 7  # 0 ~ 6

        self.set_placeholder(16, 1)

        Y_one_hot2 = tf.one_hot(self.Y, nb_classes)  # one hot op
        self.Y_one_hot_reshaped = tf.reshape(Y_one_hot2, [-1, nb_classes]) # reshape op, 리스트 [[a],[b]] -> [a, b]

        logits = self.create_layer(self.X, 16, 7, 'W', 'b')
        hypothesis = tf.nn.softmax(logits)
        self.set_hypothesis(hypothesis)

        self.set_cost_function_with_one_hot(logits, self.Y_one_hot_reshaped) #not hypothesis, but logits
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)

        prediction = tf.argmax(hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(self.Y_one_hot_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Launch graph

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for step in range(201):
            self.sess.run(self.optimizer, feed_dict={self.X: x_data, self.Y: y_data})
            if step % 100 == 0:
                loss, acc = self.sess.run([self.cost_function, accuracy], feed_dict={self.X: x_data, self.Y: y_data})
                print("Step: {:5}\tLoss: {:.5f}\tAcc: {:.7%}".format(step, loss, acc))


gildong = XXX()
gildong.run()
gildong.evaluate()

'''
# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
'''

'''
Step:     0	Loss: 5.106	Acc: 37.62%
Step:   100	Loss: 0.800	Acc: 79.21%
Step:   200	Loss: 0.486	Acc: 88.12%
Step:   300	Loss: 0.349	Acc: 90.10%
Step:   400	Loss: 0.272	Acc: 94.06%
Step:   500	Loss: 0.222	Acc: 95.05%
Step:   600	Loss: 0.187	Acc: 97.03%
Step:   700	Loss: 0.161	Acc: 97.03%
Step:   800	Loss: 0.140	Acc: 97.03%
Step:   900	Loss: 0.124	Acc: 97.03%
Step:  1000	Loss: 0.111	Acc: 97.03%
Step:  1100	Loss: 0.101	Acc: 99.01%
Step:  1200	Loss: 0.092	Acc: 100.00%
Step:  1300	Loss: 0.084	Acc: 100.00%
Step:  1400	Loss: 0.078	Acc: 100.00%
Step:  1500	Loss: 0.072	Acc: 100.00%
Step:  1600	Loss: 0.068	Acc: 100.00%
Step:  1700	Loss: 0.064	Acc: 100.00%
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
'''