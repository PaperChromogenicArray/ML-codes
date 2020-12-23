# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 22:38:16 2018

@author: Boce_Zhang
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from sklearn.model_selection import train_test_split

data = xlrd.open_workbook('combined_alldata1.xlsx')
table_data = data.sheets()[0]
data_nrows = table_data.nrows #number of rows
data_ncols = table_data.ncols #number of columns

data_datamatrix=np.zeros((data_nrows,data_ncols))

for x in range(data_ncols):
    data_cols =table_data.col_values(x)    
    
    data_cols1=np.matrix(data_cols)

    data_datamatrix[:,x]=data_cols1
    
    
species_data=np.zeros((data_nrows,1))
species_data=data_datamatrix[:,0]-1
# temperature_data=np.zeros((data_nrows,1))
# temperature_data=data_datamatrix[:,1]-1

y_species_data=tf.one_hot(species_data,6,on_value=1,off_value=None,axis=1)
# y_temperature_data=tf.one_hot(temperature_data,2,on_value=1,off_value=None,axis=1)

#y_data_tf=tf.concat([y_species_data, y_temperature_data, y_time_data, y_dye_data], 1)
#y_data_tf=tf.concat([y_species_data, y_temperature_data, y_time_data, y_dye_data], 1)
with tf.Session()as sess:
    y_data = y_species_data.eval()
    #print(sess.run(species_data))


X_data=np.zeros((data_nrows,75))
X_data=data_datamatrix[:,1:76]


x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=76)

train_nrows = len(x_train)

# Training Parameters
learning_rate = 0.000001
# convergence rate - 0.01 default - Important parameter
num_steps = 8000
# number of iteration - 20000 default
batch_size =64

display_step = 8000
examples_to_show = 10

# Network Parameters

num_input = 75 # MNIST data input (img shape: 28*28)·

num_hidden_1 =1024 # 1st layer num features
# elements per layer - 64 default - power of 2
num_code=1024
# elements per layer
num_hidden_2 =1024# 2nd layer num features (the latent dim)
# elements per layer
num_hidden_3 =1024
num_hidden_4 =1024
#alpha=0.0001
# input rows - 21 dyes for 3 RGB values per dye - 63 input
num_output = 6
#one hot classification

#beta = 0.0018  ##regulizer

train_loss=np.zeros((num_steps//10,1))
test_loss=np.zeros((num_steps//10,1))


def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.015)
    #weight square foot - adjustable - 0.1 default - important parameter
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial=tf.constant(0.3,shape=shape)
    #bias - adjustable - 0.1 default
    return tf.Variable(initial,name=name)


keep_prob = tf.placeholder(tf.float32)  # DROlP-OUT here
keep_prob = 1

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,num_input],name='x_input')
    y=tf.placeholder(tf.float32,[None,num_output],name='y_input')
with tf.name_scope('hidden_1'):
    w1=weight_variable([num_input,num_hidden_1],name='w1')
    b1=bias_variable([num_hidden_1],name='b1')
    with tf.name_scope('node_1'):
        x = tf.nn.dropout(x, keep_prob)  # DROP-OUT here
        node_1=tf.matmul(x,w1)+b1
    with tf.name_scope('relu'):
        h_1=tf.nn.relu(node_1)


with tf.name_scope('encode'):
    w2=weight_variable([num_hidden_1,num_code],name='w2')
    b2=bias_variable([num_code],name='b2')
    with tf.name_scope('sum_encode'):
        h_1 = tf.nn.dropout(h_1, keep_prob)
        sum_encode=tf.matmul(h_1,w2)+b2
    with tf.name_scope('relu'):
        h_encode=tf.nn.relu(sum_encode)

with tf.name_scope('decode'):
    w3=weight_variable([num_code,num_hidden_2],name='w3')
    b3=bias_variable([num_hidden_2],name='b3')
    with tf.name_scope('sum_decode'):
        h_encode = tf.nn.dropout(h_encode, keep_prob)
        sum_decode=tf.matmul(h_encode,w3)+b3
    with tf.name_scope('relu'):
        h_decode=tf.nn.relu(sum_decode)

with tf.name_scope('hidden_2'):
    
    w4=weight_variable([num_hidden_2,num_hidden_3],name='w4')
    b4=bias_variable([num_hidden_3],name='b4')
    with tf.name_scope('node_1'):
        h_decode = tf.nn.dropout(h_decode, keep_prob)
        node_1=tf.matmul(h_decode,w4)+b4
    with tf.name_scope('relu'):
        h_2=tf.nn.relu(node_1)

with tf.name_scope('hidden_3'):
    
    w5=weight_variable([num_hidden_3,num_hidden_4],name='w5')
    b5=bias_variable([num_hidden_4],name='b5')
    with tf.name_scope('node_1'):
        h_2 = tf.nn.dropout(h_2, keep_prob)
        node_1=tf.matmul(h_2,w5)+b5
    with tf.name_scope('relu'):
        h_3=tf.nn.relu(node_1)      

with tf.name_scope('hidden_4'):
    
    w6=weight_variable([num_hidden_4,num_output],name='w6')
    b6=bias_variable([num_output],name='b6')
    with tf.name_scope('node_1'):
        h_3 = tf.nn.dropout(h_3, keep_prob)
        node_1=tf.matmul(h_3,w6)+b6
    with tf.name_scope('relu'):
        h_4=tf.nn.relu(node_1)
    with tf.name_scope('prediction'):
        prediction=tf.nn.softmax(h_4)

with tf.name_scope('loss_mean_square'):
    #loss_mean_square=tf.reduce_mean(tf.pow(y-h_2,2))
     cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    #regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) +tf.nn.l2_loss(w4)  ##add regularizization
    #cross_entropy = tf.reduce_mean(loss + beta * regularizers)
    #tf.summary.scalar('loss_mean_square',loss_mean_square)
     tf.summary.scalar('cross',cross_entropy)
with tf.name_scope('train'):

    train_step=tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)
    # start with smaller - can go higher
    #train_step=tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
    
    
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction= tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)

    #batch_size=10
    batch_count=int(train_nrows/batch_size)
    reminder=train_nrows%batch_size
    for i in range(num_steps):
        #learning_rate = 0.1
        #if num_steps == 1000:
            #learning_rate = 0.01
        #elif num_steps == 3000:
            #learnin_rate = 0.001
        #elif num_steps == 6000:
            #learnin_rate = 0.00001
        for n in range(batch_count):
            
            train_step.run(feed_dict={x: x_train[n*batch_size:(n+1)*batch_size], y: y_train[n*batch_size:(n+1)*batch_size]})  

        if reminder>0:
            start_index = batch_count * batch_size;  
            train_step.run(feed_dict={x: x_train[start_index:train_nrows-1], y: y_train[start_index:train_nrows-1]})  
        
        iterate_accuracy = 0 
        if i%10==0:
            train_loss[i//10,0]=sess.run(accuracy,feed_dict={x:x_train,y:y_train})
            test_loss[i//10,0]=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
            print('Iter'+str(i)+', Testing Accuracy= '+str(test_loss[i//10,0])+',Training Accuracy=' +str(train_loss[i//10,0]))
 
    x_index = np.linspace(0, num_steps, num_steps//examples_to_show)
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 32,
    }
    figsize = 8,8
    figure, ax = plt.subplots(figsize=figsize)
    #figure, ax=plt.figure(figsize=(8, 8))
    
    A,=plt.plot(x_index, train_loss, color="red",label='train_accuracy',linewidth=2.0,ms=10)
    B,=plt.plot(x_index, test_loss, color="blue",label='test_accuracy',linewidth=2.0,ms=10)
    plt.legend(handles=[A,B],prop=font1)
    plt.xlabel("Iteration", font1)
    plt.ylabel("Accuracy (Species)", font1)
    
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.show()
    
    
   