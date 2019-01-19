import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# 只显示 Error

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y_real = tf.placeholder("float", [None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  # 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量

x_image = tf.reshape(x, [-1,28,28,1])  # 第一维表示图片数量即batch_size,其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_real*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   # 优化参数
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_real,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 求准确率
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(2000):
  batch_xs,batch_ys = mnist.train.next_batch(50)
  sess.run(train_step, feed_dict={x: batch_xs, y_real: batch_ys, keep_prob: 0.5})
  if i%100 == 0:
    # print("time:",i," train_accuracy:",sess.run(accuracy,feed_dict={x:batch_xs, y_real:batch_ys, keep_prob: 1.0}))   # 求训练集的准确率
    print("time:",i," test_accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels, keep_prob: 1.0}))   # 求测试集的准确率