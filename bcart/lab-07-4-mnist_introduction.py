# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from abc import abstractmethod
import mytool

class MnistSoftmax:
    X = None
    Y = None

    W = None
    b = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    def set_placeholder(self, num_of_input, num_of_neuron):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.float32, [None, num_of_neuron])

    def set_weight_bias(self, num_of_input, num_of_neuron):
        self.W = tf.Variable(tf.random_normal([num_of_input, num_of_neuron]))
        self.b = tf.Variable(tf.random_normal([num_of_neuron]))

    def set_hypothesis(self):
        # Hypothesis (using softmax)
        self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

    def set_cost_function(self):
        self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), axis=1))

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(self.cost_function)

    @abstractmethod
    def init_network(self):
        pass

    def epoch_process(self, avg_err, x_data, y_data):
        pass

    # 테스트 데이터 셋을 이용하여 평가함.
    def evaluate(self, images, labels):
        # Test model
        is_correct = tf.equal(tf.arg_max(self.hypothesis, 1), tf.arg_max(self.Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        # Test the model using test sets
        result = accuracy.eval(session=self.sess, feed_dict={self.X: db.test.images, self.Y: db.test.labels})
        #result = self.sess.run(accuracy, feed_dict={self.X: db.test.images, self.Y: db.test.labels})

        print("Accuracy: ", result)

    def learn(self, db):
        tf.set_random_seed(777)  # for reproducibility

        self.init_network()

        # parameters
        training_epochs = 3 #15
        partial_size = 100

        self.sess = tf.Session()
        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        print("Start learning")
        # Training cycle
        for epoch in range(training_epochs):
            err_4_all_data = 0
            number_of_part = int(db.train.num_examples / partial_size) #55,000 / 100

            # 처음 데이터를 100개를 읽어 최적화함.
            # 그 다음 100개 데이터에 대하여 수행.
            # 이를 모두 550번 수행하면 전체 데이터 55,000개에 대해 1번 수행하게 됨.
            # 아래 for 문장이 한번 모두 실행되면 전체 데이터에 대해 1번 실행(학습)함.
            for i in range(number_of_part):
                x_data, y_data = db.train.next_batch(partial_size)

                # 아래 에러는 일부분(100개)에 대한 것이므로 전체 에러를 구하려면 550으로 나누어주어야 함. 아래에서 수행함.
                err_4_partial, _= self.sess.run([self.cost_function, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})
                err_4_all_data += err_4_partial

            avg_err = err_4_all_data / number_of_part #

            self.epoch_process(avg_err, x_data, y_data)

        print("Ended!")

        #plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        #plt.show()

    def get_image(self, num):
        # Get one and predict
        image = db.test.images[num:num+1]
        return image

    def get_label(self, num):
        return db.test.images[num:num+1]

    def classify(self, image):
        classified = self.sess.run(tf.argmax(self.hypothesis, 1), feed_dict={self.X: image})
        return classified


class XXX (MnistSoftmax):
    def init_network (self):
        self.set_placeholder(784, 10) #28 * 28 = 784, 0~9 digits
        self.set_weight_bias(784, 10)
        self.set_hypothesis()
        self.set_cost_function()
        self.set_optimizer(0.1)

    def epoch_process(self, avg_err, x_data, y_data):
        print('Error =', '{:.9f}'.format(avg_err))


gildong = XXX()
db = mytool.load_mnist()
gildong.learn(db)
no = mytool.get_random_int(db.test.num_examples)
image = gildong.get_image(no)
label = gildong.get_label(no)
classified = gildong.classify(image)

print("Label: ", self.sess.run(tf.argmax(db.test.labels[r:r + 1], 1)))
print("Label: ", self.sess.run(tf.argmax(db.test.labels[r:r + 1], 1)))


'''
Epoch: 0001 Error = 2.827615563
Epoch: 0002 Error = 1.061300485
Epoch: 0003 Error = 0.837006755
Learning finished
Accuracy:  0.8408
Label:  [1]
Prediction:  [1]
'''