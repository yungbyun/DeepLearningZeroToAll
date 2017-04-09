import tensorflow as tf
import numpy as np

# not moldule
sess = tf.Session()

char_rdic = ['h', 'e', 'l', 'o'] # id -> char
char_dic = {w : i for i, w in enumerate(char_rdic)} # char -> id
print (char_dic)


ground_temp = 'hello'
ground_truth = [char_dic[c] for c in ground_temp]
print (ground_truth)

x_data = np.array([ [1,0,0,0],  # h
                    [0,1,0,0],  # e
                    [0,0,1,0],  # l
                    [0,0,1,0]],  # l
                     dtype='f')
# Configuration
rnn_size = len(char_dic) # 4
batch_size = 1
output_size = 4



# RNN Model
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = rnn_size,
                                       input_size = None, # deprecated at tensorflow 0.9
                                       #activation = tanh,
                                       )

print(rnn_cell)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
print(initial_state)



initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size]) #  위 코드와 같은 결과
print(initial_state_1)


print (x_data)
x_split = tf.split(x_data, rnn_size, 0) # 가로축으로 ?개로 split
#
# sess.run(x_split)
# sess.run(x_split[0])
print(x_split)

#outputs, state = tf.nn.rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)   #구버전
outputs, state = tf.contrib.rnn.static_rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)
sess.run(tf.initialize_all_variables())
print (outputs)
print (state)
print(sess.run(state))
logits = tf.reshape(tf.concat(outputs,1), # shape = 1 x 16 버전업으로 파라미터 순서바뀜
                    [-1,rnn_size])        # shape = 4 x 4
logits.get_shape()

targets = tf.reshape(ground_truth[1:], [-1]) # a shape of [-1] flattens into 1-D\
targets.get_shape()




weights = tf.ones([len(char_dic) * batch_size])
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
sess.run(tf.initialize_all_variables())
for i in range(500):
    sess.run(train_op)
    result = sess.run(tf.argmax(d3_logics[0], 1))
    if i% 20 == 0:
        print(sess.run(cost))
        print(result, [char_rdic[t] for t in result])
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

print(sess.run(state)[0, 0] + sess.run(outputs)[0])