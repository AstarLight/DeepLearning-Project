#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定使用的GPU的ID

'''
脚本功能：搭建神经网络+tensorboard可视化
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') #权值采取随机生成的初始化
            tf.histigram_summary(layer_name+'weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs
    
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')


#输入层
layer1 = add_layer(xs,1,10,n_layer = 1,activation_function=tf.nn.relu)
#隐藏层
prediction = add_layer(layer1,10,1,n_layer = 2,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                        reduction_indices=[1]))
    tf.scalar_summary('loss',loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#init = tf.initialize_all_variables() ##deprecated
init = tf.global_variables_initializer()
sess = tf.Session()
#tensorboard可视化：tensorboard --logdir='./logs'
writer = tf.summary.FileWriter('logs/',sess.graph)  
sess.run(init)

#可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()  #使得plt.show()之后不停止，继续往下执行
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        try:
            ax.lines.remove(lines[0]) 
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)      
        plt.pause(0.1) #暂停0.1秒

                    

                    

