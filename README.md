# DeepFM-Guide
从零开始运行DeepFM项目。
## 实验准备
1. 安装pycharm
2. 安装python
## tensorflow基础知识
1.tensorflow简介
TensorFlow是采用数据流图（data　flow　graphs）来计算, 所以首先我们得创建一个数据流流图,然后再将我们的数据（数据以张量(tensor)的形式存在）放在数据流图中计算. 节点（Nodes）在图中表示数学操作,图中的边（edges）则表示在节点间相互联系的多维数据数组, 即张量（tensor).训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来.
**张量**（Tensor):张量有多种. 零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 [1],一阶张量为 向量 (vector), 比如 一维的 [1, 2, 3],二阶张量为 矩阵 (matrix), 比如 二维的 [[1, 2, 3],[4, 5, 6],[7, 8, 9]],以此类推, 还有 三阶 三维的 …
2.从一个例子讲起
首先，我们来看一个简单的例子：
```python
import tensorflow as tf
import numpy as np
#tensorflow中大部分数据是float32

#create real data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###

#定义变量
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

#如何计算预测值
y = Weights * x_data + biases

# loss function
loss = tf.reduce_mean(tf.square(y-y_data))

#梯度下降优化器，定义learning rate
optimizer = tf.train.GradientDescentOptimizer(0.5)

#训练目标是loss最小化
train = optimizer.minimize(loss)

#初始化变量，即初始化 Weights 和 biases
init = tf.global_variables_initializer()

#创建session，进行参数初始化
sess = tf.Session()
sess.run(init)

#开始训练200步，每隔20步输出一下两个参数
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
### create tensorflow structure end ###
```
在上面的例子中，我们想要预测的方程式y=0.1*x + 0.3,给定训练样本，通过梯度下降法来预测参数W和偏置b，我们使用numpy生成了我们的训练数据：
```python
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
```
随后，我们使用tf.Variable定义了我们的变量Weights和biases(以下简称w和b），Weights通过一个均匀分布随机产生，而bias则设置为0，同时二者的形状均为1维，因为只有一个数：
```python
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
```
好了，有了变量，我们想要学习w和b，只需要用训练数据x来得到预测值，最小化预测值和实际值的差距就好，所以，我们定义了损失函数为平方损失函数，并通过0.5学习率的梯度下降法来进行参数调整：
```python
#如何计算预测值
y = Weights * x_data + biases

# loss function
loss = tf.reduce_mean(tf.square(y-y_data))

#梯度下降优化器，定义learning rate
optimizer = tf.train.GradientDescentOptimizer(0.5)

#训练目标是loss最小化
train = optimizer.minimize(loss)
```
在tf中定义的变量都需要经过初始化的操作，所以我们定义了一个初始化变量的操作：
```python
#初始化变量，即初始化 Weights 和 biases
init = tf.global_variables_initializer()
```
接下来我们就可以开始训练了，训练必须创建一个session，通过run方法对指定的节点进行训练，这里一定要注意先要对参数进行初始化，否则后面是无法开始训练的。想要观察训练过程中的参数变化的话，也需要通过run方法：
```python
#创建session，进行参数初始化
sess = tf.Session()
sess.run(init)

#开始训练200步，每隔20步输出一下两个参数
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
```
这里 我们直接run的是train这一步，想要运行这一步，必须先得到optimizier和loss，想要得到loss就要得到预测值....依次往前推，所以run(train)实际上就是对整个tensor流图的训练。
好啦，说了这么多，我们来看一下我们的输出结果吧：
```python
0 [ 0.65090138] [-0.04130311]
20 [ 0.23774943] [ 0.21987261]
40 [ 0.13388598] [ 0.2802889]
60 [ 0.10833587] [ 0.29515111]
80 [ 0.10205062] [ 0.2988072]
100 [ 0.10050445] [ 0.29970658]
120 [ 0.10012411] [ 0.29992783]
140 [ 0.10003054] [ 0.29998225]
160 [ 0.10000751] [ 0.29999563]
180 [ 0.10000186] [ 0.29999894]
200 [ 0.10000047] [ 0.29999974]
```
可以看到，经过200步，准确的说在80步左右的时候，我们的tensorflow已经能够很准确的将Weights和Bias学习出来了。
3.tf.Session 
Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分，有两种使用Session的方式，我们可以从下面的例子中看出来,但在实际中，我们更推荐后者：
```python
import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()


with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
```
4.tf.Variable
在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。定义语法： state = tf.Variable().如果你在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 init = tf.global_variables_initializer().到这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.
```python
import tensorflow as tf

#定义变量，给定初始值和name
state = tf.Variable(0,name="counter")
#counter:0
print(state.name)

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

#这里只是定义，必须用session.run来执行
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
```
5.TF placeholder
placeholder 是 Tensorflow 中的占位符，暂时储存变量.
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
```python
import tensorflow as tf

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3.],input2:[5]}))
```
参考：
链接：https://www.jianshu.com/p/ce213e6b2dc0
