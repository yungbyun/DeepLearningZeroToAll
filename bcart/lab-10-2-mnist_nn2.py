# Lab 10 MNIST and NN
from mytype import MyType
from mnist_neural_network import MnistNeuralNetwork


class XXX (MnistNeuralNetwork):
    def init_network(self):
        self.set_placeholder(784, 10)
        Wa, ba, outputa = self.create_layer(None, 784, 256, MyType.RELU)
        Wb, bb, outputb = self.create_layer(outputa, 256, 256, MyType.RELU)
        Wc, bc,    hypo = self.create_layer(outputb, 256, 10, MyType.LINEAR)
        self.set_hypothesis(hypo)
        self.set_cost_function(MyType.SOFTMAX_LOGITS)
        self.set_optimizer(MyType.ADAM, 0.001)


gildong = XXX()
gildong.learn(15, 100)
gildong.evaluate()
gildong.classify_random_image()
gildong.show_error()

'''
Epoch: 0001 cost = 146.360605907
Epoch: 0002 cost = 40.355669494
Epoch: 0003 cost = 25.141711207
Learning Finished!
Accuracy: 0.91
Label:  [4]
Prediction:  [4]
'''

'''
Epoch: 0013 cost = 0.820965160
Epoch: 0014 cost = 0.624131458
Epoch: 0015 cost = 0.454633765
Learning Finished!
Accuracy: 0.9455
'''
