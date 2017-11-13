#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定使用的GPU的ID

'''
脚本功能：拟合直线，找出y=ax+b的a和b
'''

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#Weights随机生成（-1.0到1.0的范围内生成），baises初始化为0
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

#激活神经网络
sess = tf.Session()
sess.run(init)

#训练2000次
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step,sess.run(Weights),sess.run(biases))
