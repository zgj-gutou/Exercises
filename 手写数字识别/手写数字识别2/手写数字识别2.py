# coding=utf-8

import tensorflow as tf
# 下载MNIST数据集到'MNIST_data'文件夹并解压
# 为了方便起见，我们已经准备了一个脚本来自动下载和导入MNIST数据集。它会自动创建一个'MNIST_data'的目录来存储数据。（这是官方的程序）
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 设置权重weights和偏置biases作为优化变量，初始值设为0
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

# 构建模型
x = tf.placeholder("float", [None, 784])
y = tf.nn.softmax(tf.matmul(x, weights) + biases)                                   # 模型的预测值
y_real = tf.placeholder("float", [None, 10])                                        # 真实值

cross_entropy = -tf.reduce_sum(y_real * tf.log(y))                                  # 预测值与真实值的交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)        # 使用梯度下降优化器最小化交叉熵

# 开始训练
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                                # 每次随机选取100个数据进行训练，即所谓的“随机梯度下降（Stochastic Gradient Descent，SGD）”
    sess.run(train_step, feed_dict={x: batch_xs, y_real:batch_ys})                  # 正式执行train_step，用feed_dict的数据取代placeholder

    if i % 100 == 0:
        # 每训练100次后评估模型
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_real, 1))       # 比较预测值和真实值是否一致
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))             # 统计预测正确的个数，取均值得到准确率
        print("times:",i," accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))