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

参考：
链接：https://www.jianshu.com/p/ce213e6b2dc0
