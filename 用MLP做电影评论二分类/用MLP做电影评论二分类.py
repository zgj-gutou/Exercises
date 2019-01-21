from keras.datasets import imdb

'''
train_data 和 test_data 这两个变量都是评论组成的列表，每条评论又是单词索引组成
的列表（表示一系列单词）。 train_labels 和 test_labels 都是 0 和 1 组成的列表，其中 0
代表负面（negative），1 代表正面（positive）。
'''
# 参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词。低频单词将被舍弃。这样得到的向量数据不会太大，便于处理
# 由于限定为前 10 000 个最常见的单词，单词索引都不会超过 10 000。最大为9999
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()  # word_index是一个将单词映射为整数索引的字典
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])  # 键值颠倒，将整数索引映射为单词
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
# print(decoded_review)

# 准备数据
# onehot编码，将数据向量化，转为二进制矩阵
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))  # 创建一个形状为 (len(sequences),dimension) 的零矩阵
    for i,sequence in enumerate(sequences):
        # print("i:",i)
        # print("sequence:",sequence)
        results[i,sequence] = 1   # 第i行，sequence的数字对应的列，设为1。也就是result[i]的指定索引设为1
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 将标签向量化
y_train = np.asarray(train_labels).astype("float32")   # 其实这里只是转换了一下数据类型而已，把int转换为了float32
y_test = np.asarray(test_labels).astype("float32")

# 模型定义
from keras import models
from keras import layers

model = models.Sequential() # 开始 Keras 序列模型
# 传入 Dense 层的参数（16）是该层隐藏单元的个数。一个隐藏单元（hidden unit）是该层表示空间的一个维度
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# 编译模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

'''
现在使用 512 个样本组成的小批量，将模型训练 20 个轮次（即对 x_train 和 y_train 两
个张量中的所有样本进行 20 次迭代）。与此同时，你还要监控在留出的 10 000 个样本上的损失
和精度。你可以通过将验证数据传入 validation_data 参数来完成。
'''
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# 注意，调用 model.fit() 返回了一个 History 对象。这个对象有一个成员 history ，它是一个字典，包含训练过程中的所有数据。
# 字典中包含 4 个条目，对应训练过程和验证过程中监控的指标。
print(history.history.keys())

# 绘制训练损失和验证损失。请注意，由于网络的随机初始化不同，你得到的结果可能会略有不同。
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,"bo",label = "Training loss")
plt.plot(epochs,val_loss_values,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label="Training acc")
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title("Training and Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

'''
history = model.fit(x_train,y_train,epochs = 4,batch_size=512)
results = model.evaluate(x_test,y_test)
print("results:",results)
print("predict:",model.predict(x_test))
'''


