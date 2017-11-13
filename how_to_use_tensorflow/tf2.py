#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定使用的GPU的ID

'''
脚本功能：拟合直线，找出y=ax+b的a和b
'''

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

#矩阵乘法。numpy矩阵乘法：np.dot(m1,m2s)
product = tf.matmul(matrix1,matrix2)

'''
session练习
'''
#method 1:显式close()
sess = tf.Session()
result = sess.run(product)
print result
sess.close()


#method 2:这种方法不需要显式sess.clsoe()
with tf.Session() as sess:
    result2 = sess.run(product)
    print result2
 

'''
Variables练习
'''
state = tf.Variable(0,name='counter') 
print state.name
one = tf.Variable(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value) #将new_value的值更新到state

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)



'''
placeholdor练习
'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print sess.run(output,feed_dict={input1:[7.],input2:[2.]}) #在执行这条语句时input1，input2具体值才被确定






    
    

    

  
