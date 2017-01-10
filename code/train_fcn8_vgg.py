#!/usr/bin/env python
import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn_vgg
import utils
import input

from tensorflow.python.framework import ops


sess = tf.InteractiveSession()
batch_size = 2
num_classes = 18
x = tf.placeholder("float", [batch_size, 100, 150, 3])
y_ = tf.placeholder(tf.int32, [batch_size, 100, 150, 1])
train_mode = tf.placeholder(tf.bool) 

vgg_npy_path = ''
vgg_fcn = fcn_vgg.VGG(trainable=True)
vgg_fcn.build(rgb=x, train_mode=train_mode)

y = vgg_fcn.pred_up 

logits = tf.reshape(y, (-1, num_classes))
epsilon = tf.constant(value=1e-4)
logits = logits + epsilon
labels = tf.reshape(y_, (-1, num_classes))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

width = 100
height = 150
#zero = tf.constant(0.0, dtype=tf.int32, shape=[batch_size,])
one = tf.constant(1, dtype=tf.int32, shape=[batch_size, width, height, 1])
two = tf.constant(2, dtype=tf.int32, shape=[batch_size, width, height, 1])
three = tf.constant(3, dtype=tf.int32, shape=[batch_size, width, height, 1])
four = tf.constant(4, dtype=tf.int32, shape=[batch_size, width, height, 1])
five = tf.constant(5, dtype=tf.int32, shape=[batch_size, width, height, 1])
six = tf.constant(6, dtype=tf.int32, shape=[batch_size, width, height, 1])
seven = tf.constant(7, dtype=tf.int32, shape=[batch_size, width, height, 1])
eight = tf.constant(8, dtype=tf.int32, shape=[batch_size, width, height, 1])
nine = tf.constant(9, dtype=tf.int32, shape=[batch_size, width, height, 1])
ten = tf.constant(10, dtype=tf.int32, shape=[batch_size, width, height, 1])
eleven = tf.constant(11, dtype=tf.int32, shape=[batch_size, width, height, 1])
twelve = tf.constant(12, dtype=tf.int32, shape=[batch_size, width, height, 1])
thirteen = tf.constant(13, dtype=tf.int32, shape=[batch_size, width, height, 1])
fourteen = tf.constant(14, dtype=tf.int32, shape=[batch_size, width, height, 1])
fifteen = tf.constant(15, dtype=tf.int32, shape=[batch_size, width, height, 1])
sixteen = tf.constant(16, dtype=tf.int32, shape=[batch_size, width, height, 1])
seventeen = tf.constant(17, dtype=tf.int32, shape=[batch_size, width, height, 1])

single_val = tf.constant(2, dtype=tf.float32, shape=[batch_size,])

one_pred = tf.equal(one, y)
one_truth = tf.equal(one, y_)
one_intersect = tf.equal(one_pred, one_truth)
one_IU = tf.div(tf.cast(((tf.reduce_sum(one_intersect) - tf.reduce_sum(one) + tf.reduce_sum(one_truth)+tf.reduce_sum(one_pred)) / (tf.reduce_sum(one_truth)+tf.reduce_sum(one_pred))), tf.float32), single_val)

two_pred = tf.equal(two, y)
two_truth = tf.equal(two, y_)
two_intersect = tf.equal(two_pred, two_truth)
two_IU = tf.div(tf.cast(((tf.reduce_sum(two_intersect) - tf.reduce_sum(two) + tf.reduce_sum(two_truth)+tf.reduce_sum(two_pred)) / (tf.reduce_sum(two_truth)+tf.reduce_sum(two_pred))), tf.float32), single_val)

three_pred = tf.equal(three, y)
three_truth = tf.equal(three, y_)
three_intersect = tf.equal(three_pred, three_truth)
three_IU = tf.div(tf.cast(((tf.reduce_sum(three_intersect) - tf.reduce_sum(three) + tf.reduce_sum(three_truth)+tf.reduce_sum(three_pred)) / (tf.reduce_sum(three_truth)+tf.reduce_sum(three_pred))), tf.float32), single_val)

four_pred = tf.equal(four, y)
four_truth = tf.equal(four, y_)
four_intersect = tf.equal(four_pred, four_truth)
four_IU = tf.div(tf.cast(((tf.reduce_sum(four_intersect) - tf.reduce_sum(four) + tf.reduce_sum(four_truth)+tf.reduce_sum(four_pred)) / (tf.reduce_sum(four_truth)+tf.reduce_sum(four_pred))), tf.float32), single_val)

five_pred = tf.equal(five, y)
five_truth = tf.equal(five, y_)
five_intersect = tf.equal(five_pred, five_truth)
five_IU = tf.div(tf.cast(((tf.reduce_sum(five_intersect) - tf.reduce_sum(five) + tf.reduce_sum(five_truth)+tf.reduce_sum(five_pred)) / (tf.reduce_sum(five_truth)+tf.reduce_sum(five_pred))), tf.float32), single_val)

six_pred = tf.equal(six, y)
six_truth = tf.equal(six, y_)
six_intersect = tf.equal(six_pred, six_truth)
six_IU = tf.div(tf.cast(((tf.reduce_sum(six_intersect) - tf.reduce_sum(six) + tf.reduce_sum(six_truth)+tf.reduce_sum(six_pred)) / (tf.reduce_sum(six_truth)+tf.reduce_sum(six_pred))), tf.float32), single_val)

seven_pred = tf.equal(seven, y)
seven_truth = tf.equal(seven, y_)
seven_intersect = tf.equal(seven_pred, seven_truth)
seven_IU = tf.div(tf.cast(((tf.reduce_sum(seven_intersect) - tf.reduce_sum(seven) + tf.reduce_sum(seven_truth)+tf.reduce_sum(seven_pred)) / (tf.reduce_sum(seven_truth)+tf.reduce_sum(seven_pred))), tf.float32), single_val)

eight_pred = tf.equal(eight, y)
eight_truth = tf.equal(eight, y_)
eight_intersect = tf.equal(eight_pred, eight_truth)
eight_IU = tf.div(tf.cast(((tf.reduce_sum(eight_intersect) - tf.reduce_sum(eight) + tf.reduce_sum(eight_truth)+tf.reduce_sum(eight_pred)) / (tf.reduce_sum(eight_truth)+tf.reduce_sum(eight_pred))), tf.float32), single_val)

nine_pred = tf.equal(nine, y)
nine_truth = tf.equal(nine, y_)
nine_intersect = tf.equal(nine_pred, nine_truth)
nine_IU = tf.div(tf.cast(((tf.reduce_sum(nine_intersect) - tf.reduce_sum(nine) + tf.reduce_sum(nine_truth)+tf.reduce_sum(nine_pred)) / (tf.reduce_sum(nine_truth)+tf.reduce_sum(nine_pred))), tf.float32), single_val)

ten_pred = tf.equal(ten, y)
ten_truth = tf.equal(ten, y_)
ten_intersect = tf.equal(ten_pred, ten_truth)
ten_IU = tf.div(tf.cast(((tf.reduce_sum(ten_intersect) - tf.reduce_sum(ten) + tf.reduce_sum(ten_truth)+tf.reduce_sum(ten_pred)) / (tf.reduce_sum(ten_truth)+tf.reduce_sum(ten_pred))), tf.float32), single_val)

eleven_pred = tf.equal(eleven, y)
eleven_truth = tf.equal(eleven, y_)
eleven_intersect = tf.equal(eleven_pred, eleven_truth)
eleven_IU = tf.div(tf.cast(((tf.reduce_sum(eleven_intersect) - tf.reduce_sum(eleven) + tf.reduce_sum(eleven_truth)+tf.reduce_sum(eleven_pred)) / (tf.reduce_sum(eleven_truth)+tf.reduce_sum(eleven_pred))), tf.float32), single_val)


twelve_pred = tf.equal(twelve, y)
twelve_truth = tf.equal(twelve, y_)
twelve_intersect = tf.equal(twelve_pred, twelve_truth)
twelve_IU = tf.div(tf.cast(((tf.reduce_sum(twelve_intersect) - tf.reduce_sum(twelve) + tf.reduce_sum(twelve_truth)+tf.reduce_sum(twelve_pred)) / (tf.reduce_sum(twelve_truth)+tf.reduce_sum(twelve_pred))), tf.float32), single_val)

thirteen_pred = tf.equal(thirteen, y)
thirteen_truth = tf.equal(thirteen, y_)
thirteen_intersect = tf.equal(thirteen_pred, thirteen_truth)
thirteen_IU = tf.div(tf.cast(((tf.reduce_sum(thirteen_intersect) - tf.reduce_sum(thirteen) + tf.reduce_sum(thirteen_truth)+tf.reduce_sum(thirteen_pred)) / (tf.reduce_sum(thirteen_truth)+tf.reduce_sum(thirteen_pred))), tf.float32), single_val)

fourteen_pred = tf.equal(fourteen, y)
fourteen_truth = tf.equal(fourteen, y_)
fourteen_intersect = tf.equal(fourteen_pred, fourteen_truth)
fourteen_IU = tf.div(tf.cast(((tf.reduce_sum(fourteen_intersect) - tf.reduce_sum(fourteen) + tf.reduce_sum(fourteen_truth)+tf.reduce_sum(fourteen_pred)) / (tf.reduce_sum(fourteen_truth)+tf.reduce_sum(fourteen_pred))), tf.float32), single_val)

fifteen_pred = tf.equal(fifteen, y)
fifteen_truth = tf.equal(fifteen, y_)
fifteen_intersect = tf.equal(fifteen_pred, fifteen_truth)
fifteen_IU = tf.div(tf.cast(((tf.reduce_sum(fifteen_intersect) - tf.reduce_sum(fifteen) + tf.reduce_sum(fifteen_truth)+tf.reduce_sum(fifteen_pred)) / (tf.reduce_sum(fifteen_truth)+tf.reduce_sum(fifteen_pred))), tf.float32), single_val)

sixteen_pred = tf.equal(sixteen, y)
sixteen_truth = tf.equal(sixteen, y_)
sixteen_intersect = tf.equal(sixteen_pred, sixteen_truth)
sixteen_IU = tf.div(tf.cast(((tf.reduce_sum(sixteen_intersect) - tf.reduce_sum(sixteen) + tf.reduce_sum(sixteen_truth)+tf.reduce_sum(sixteen_pred)) / (tf.reduce_sum(sixteen_truth)+tf.reduce_sum(sixteen_pred))), tf.float32), single_val)


seventeen_pred = tf.equal(seventeen, y)
seventeen_truth = tf.equal(seventeen, y_)
seventeen_intersect = tf.equal(seventeen_pred, seventeen_truth)
seventeen_IU = tf.div(tf.cast(((tf.reduce_sum(seventeen_intersect) - tf.reduce_sum(seventeen) + tf.reduce_sum(seventeen_truth)+tf.reduce_sum(seventeen_pred)) / (tf.reduce_sum(seventeen_truth)+tf.reduce_sum(seventeen_pred))), tf.float32), single_val) 


tf.scalar_summary('Loss', cost)
merged = tf.merge_all_summaries()
log_dir = 'fcn'
train_writer = tf.train.SummaryWriter(log_dir+'/train')
val_writer = tf.train.SummaryWriter(log_dir+'/val')
# train
n_epoch = 30
global_step = tf.Variable(0)
starter_learning_rate = 0.00005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
print_freq = 1

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, global_step=global_step)
#train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)

sess.run(tf.initialize_all_variables())

iter_show = 0
for epoch in range(n_epoch):
    start_time = time.time()
    iter_per_epoch = 1635
    for iter in xrange(iter_per_epoch):
        x_batch, y_batch = input.load_batchsize_images('train', batch_size)
        feed_dict = {x: x_batch, y_: y_batch}
        #conv1, conv2, conv3, conv4, conv5, fc8, fc7, fc6, pool3 = sess.run([network.conv1, network.conv2, network.conv3, network.conv4, network.conv5, network.fc8, network.fc7, network.fc6, network.pool3], feed_dict=feed_dict)
        _, err, lr, train_summary = sess.run([train_op, cost, learning_rate, merged], feed_dict=feed_dict)
        iter_show += 1
        train_writer.add_summary(train_summary, iter_show)
        #network.save_npy(sess=sess, npy_path="test_save.npy")

        if iter % 10 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Train Loss: %f" % err)

        x_val_batch, y_val_batch = input.load_batchsize_images('validation', batch_size)
        feed_dict_val = {x: x_val_batch, y_: y_val_batch}
        val_err, val_summary = sess.run([cost, merged], feed_dict=feed_dict_val)
        val_writer.add_summary(val_summary, iter_show)  
        if iter % 10 == 0:      
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, iter+1, lr))
            print("   Val Loss: %f" % val_err)


    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        test_loss, batch_test_size = 0, 2
        test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven, test_twelve, test_thirteen, test_fourteen, test_fifteen, test_sixteen, test_seventeen = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        test_iters = 312
        for iter_test in xrange(test_iters):
            #print iter_test
            x_test_batch, y_test_batch = input.load_batchsize_images('test', batch_test_size)
            feed_dict = {x: x_test_batch, y_: y_test_batch}

            tensors = [vgg_fcn.pred_up]
            up = sess.run(tensors, feed_dict=feed_dict)
            up_color = utils.color_image(up[0])
            scp.misc.imsave(str(epoch+1)+'_'+str(iter_test+1)+'fcn8.png', up_color)

            err, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen = sess.run([cost, one_IU, two_IU, three_IU, four_IU, five_IU, six_IU, seven_IU, eight_IU, nine_IU, ten_IU, eleven_IU, twelve_IU, thirteen_IU, fourteen_IU, fifteen_IU, sixteen_IU, seventeen_IU], feed_dict=feed_dict)
            test_loss += err
            test_one += one; test_two += two; test_three += three; test_four += four 
            test_five += five; test_six += six; test_seven += seven;
            test_eight += eight; test_nine += nine; test_ten += ten;
            test_eleven += eleven; test_twelve += twelve; test_thirteen += thirteen;
            test_fourteen += fourteen; test_fifteen += fifteen; test_sixteen += sixteen;
            test_seventeen += seventeen

        print("   Test Loss: %f" % (test_loss/ test_iters))
        with open('test_result.txt','a') as w_f:
            w_f.write(str(epoch)+'\t'+str(test_loss/test_iters)+'\n')
            w_f.write(str(epoch)+'\t'+str(test_one/test_iters)+'\t'+str(test_two/test_iters)+'\t'+str(test_three/test_iters)+'\t'+str(test_four/test_iters)+'\t'+str(test_five/test_iters)+'\t'
                +str(test_six/test_iters)+'\t'+str(test_seven/test_iters)+'\t'+str(test_eight/test_iters)+'\t'+str(test_nine/test_iters)+'\t'+str(test_ten/test_iters)+'\t'+str(test_eleven/test_iters)+'\t'+
                str(test_twelve/test_iters)+'\t'+str(test_thirteen/test_iters)+'\t'+str(test_fourteen/test_iters)+'\t'+str(test_fifteen/test_iters)+'\t'+str(test_sixteen/test_iters)+'\t'+str(test_seventeen/test_iters)+'\n')
        w_f.close()
        

train_writer.close()
val_writer.close()
