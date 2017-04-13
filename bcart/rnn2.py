import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp

class XXX:


    def run(self):
        # not moldule
        sess = tf.Session()

        char_dic = ['h', 'e', 'l', 'o']
        char_index = {'h': 0, 'e': 1, 'l': 2, 'o': 3}

        hello_index = [0, 1, 2, 2, 3]

        x_data = np.array([[1,0,0,0],  # h
                           [0,1,0,0],  # e
                           [0,0,1,0],  # l
                           [0,0,1,0]],  # l
                           dtype='f')

        # Configuration
        rnn_size = 4 #len(char_index), hidden_size, output_dim
        batch_size = 1 # 배치, 시퀀스, 입력

        # RNN Model
        rnn_cell = rnn.BasicLSTMCell(num_units=4, state_is_tuple=True) #히든
        initial_state = rnn_cell.zero_state(batch_size, tf.float32)
        print(initial_state) # shape=(1,4) [[a,b,c,d]]

        x_split = tf.split(x_data, rnn_size, 0) # 가로축으로 ?개로 split하는 op

        #outputs, state = tf.nn.rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)   #구버전
        outputs, state = tf.contrib.rnn.static_rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)

        sess.run(tf.global_variables_initializer())

        print('\n')
        pp.pprint (sess.run(outputs))

        print('\n')
        pp.pprint (sess.run(state)) #c(4), h(4)

        # shape = 1 x 16 버전업으로 파라미터 순서바뀜, shape = 4 x 4
        logits = tf.reshape(tf.concat(outputs,1), [-1,rnn_size])
        logits.get_shape()

        targets = tf.reshape(hello_index[1:], [-1]) # a shape of [-1] flattens into 1-D\
        targets.get_shape()
        print('Target:', targets.get_shape())

        weights = tf.ones([len(char_index) * batch_size])
        # seq2seq.sequence_loss의 오류검출 코드에서 리스트는 안되고 꼭 텐서로 해야 된다고 합니다.
        # 하지만 2,1,1차원 변수를 함수가 요구하는 3,2,2차원 변수로 만들기 위해 이렇게 조정해줍니다.
        d3_logics = tf.Variable([logits])
        d2_targets = tf.Variable([targets])
        d2_weights = tf.Variable([weights])
        #loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights]) <<원랜 이게 됐었다고 합니다.
        loss = tf.contrib.seq2seq.sequence_loss(d3_logics, d2_targets, d2_weights)
        cost = tf.reduce_sum(loss) / batch_size
        train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

        # Launch the graph in a session
        sess.run(tf.global_variables_initializer())

        for i in range(100): #500
            sess.run(train_op)
            result = sess.run(tf.argmax(d3_logics[0], 1))
            if i% 50 == 0:
                print(sess.run(cost))
                print(result, [char_dic[t] for t in result])
        print(sess.run(d2_weights))



        def forward_propagation(str):
            # The total number of time steps
            T = len(str)
            # During forward propagation we save all hidden states in s because need them later.
            # We add one additional element for the initial hidden, which we set to 0
            s = np.zeros((T + 1, 2))
            s[-1] = np.zeros(2)
            # The outputs at each time step. Again, we save them for later.
            o = np.zeros((T,10))
            # For each time step...
            for t in np.arange(T):
                # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
                s[t] = np.tanh(str [:,str[t]] +d2_weights.dot(s[t-1]))
                o[t] = tf.nn.softmax(V.dot(s[t]))
            return [o, s]

        print(sess.run(state))
        print(sess.run(outputs))

        print(sess.run(state)[0])
        print(sess.run(outputs)[0])


gildong = XXX()
gildong.run()

