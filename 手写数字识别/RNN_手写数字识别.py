# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
His code is a very good one for RNN beginners. Feel free to check it out.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
# batch_size = 128
batch_size = 256

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {   # 对权重进行随机初始化
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {   # 对偏差进行随机初始化
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    # transpose the inputs shape from
    # X ==> (256 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (256 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']   # 这里的维度是（256*28,128)，输入先经过一个线性变化
    # X_in ==> (256 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])   # 这里的维度是（256,28,128）

    # cell
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)  # 一个lstm单元的输入是向量，所以有这么多隐藏单元n_hidden_units
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)   # 初始状态的维度跟batch_size和隐藏单元个数有关，这里shape=(256, 128)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']    # 可以这样用隐状态final_state[1]求results,也可以用下面的方法即最后时刻的输出outputs求results

    # # or
    # unpack to list [(batch, outputs)..] * steps
    # 这里的outputs是[batch_size, max_time, cell.output_size(即hidden_units)]的形式，现在要取最后一个时间的outputs,所以要调换一下max_time和batch_size，这样就可以直接用outputs[-1]来得到最后一个的输出
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))  # tf.unstack默认是按行分解。
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)  # outputs[-1]是指最后一个时间的输出，outputs[-1]再经过一个线性变化得到结果

    return results

pred = RNN(x, weights, biases)  # 此时x还没有数值，后面用feed_dict输入
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # 此时y还没有数值，后面用feed_dict输入
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # print(len(batch_xs))
        # print(len(batch_xs[0]))
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])  # 把[batch_size,784]变形为[batch_size,28,28],这样才符合x的维度
        sess.run(train_op, feed_dict={x: batch_xs,y: batch_ys,})
        if step % 20 == 0:
            batch_xs, batch_ys = mnist.test.next_batch(batch_size)
            # 取batch_size大小的测试集，因为初始状态跟batch_size有关，所以这里取batch_size大小的测试集数据，否则会报错。也可以把batch_size作为一个参数来传递
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            print("time:",step," accuracy:",sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,}))
        step += 1