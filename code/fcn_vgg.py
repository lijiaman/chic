import os
import tensorflow as tf
import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG:
    """
    An implementation of VGG model.
    """

    def __init__(self, vgg_npy_path=None, trainable=True):
        if vgg_npy_path is not None:
            self.data_dict = np.load(vgg_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb, flag=None, reuse=None, train_mode=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        with tf.variable_scope(flag, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()


            rgb_scaled = rgb * 255.0
            width = 100
            height = 150

	        # Convert RGB to BGR
            red, green, blue = tf.split(3, 3, rgb_scaled)
            assert red.get_shape().as_list()[1:] == [width, height, 1]
            assert green.get_shape().as_list()[1:] == [width, height, 1]
            assert blue.get_shape().as_list()[1:] == [width, height, 1]
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
                ])

            assert bgr.get_shape().as_list()[1:] == [width, height, 3]

            self.conv1_1 = self.conv_layer(bgr, 3, 64, 3, 1, "conv1_1", padding="VALID")
            self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, 3, 1, "conv1_2", pad=[1,1])
            self.pool1 = self.max_pool(self.conv1_2, 2, 2, "pool1")
            #print "pool1.shape:"
            #print self.pool1.get_shape()

            self.conv2_1 = self.conv_layer(self.pool1, 64, 128, 3, 1, "conv2_1", pad=[1, 1])
            self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, 3, 1, "conv2_2", pad=[1, 1])
            self.pool2 = self.max_pool(self.conv2_2, 2, 2, "pool2")
            #print "pool2.shape:"
            #print self.pool2.get_shape()

            self.conv3_1 = self.conv_layer(self.pool2, 128, 256, 3, 1, "conv3_1", pad=[1, 1])
            self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, 3, 1, "conv3_2", pad=[1, 1])
            self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, 3, 1, "conv3_3", pad=[1, 1])
            self.pool3 = self.max_pool(self.conv3_3, 2, 2, "pool3")

            self.conv4_1 = self.conv_layer(self.pool3, 256, 512, 3, 1, "conv4_1", pad=[1, 1])
            self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, 3, 1, "conv4_2", pad=[1, 1])
            self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, 3, 1, "conv4_3", pad=[1, 1])
            self.pool4 = self.max_pool(self.conv4_3, 2, 2, "pool4")

            self.conv5_1 = self.conv_layer(self.pool4, 512, 512, 3, 1, "conv5_1", pad=[1, 1])
            self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, 3, 1, "conv5_2", pad=[1, 1])
            self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, 3, 1, "conv5_3", pad=[1, 1])
            self.pool5 = self.max_pool(self.conv5_3, 2, 2, "pool5")

            self.fc6 = self.conv_layer(self.pool5, 512, 4096, 3, 1, "fc6_conv")#origin size is k_size=7

            if train_mode is not None:
                self.fc6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.fc6, 0.5), lambda: self.fc6)
            elif self.trainable:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5)

            self.fc7 = self.conv_layer(self.fc6, 4096, 4096, 1, 1, "fc7_conv")
            if train_mode is not None:
                self.fc7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.fc7, 0.5), lambda: self.fc7)
            elif self.trainable:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5)

            self.score_fr = self.conv_layer(self.fc7, 4096, 18, 1, 1, "score_fr")
            self.upscore2 = self.deconv_layer(self.score_fr, 18, 18, 2, 2, "upscore2")
            self.score_pool4 = self.conv_layer(self.pool4, 512, 18, 1, 1, "score_pool4")
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

            self.upscore_pool4 = self.deconv_layer(self.fuse_pool4, 18, 18, 1, 2, "upscore_pool4")
            self.score_pool3 = self.conv_layer(self.pool3, 256, 18, 1, 1, "score_pool3")
            self.fuse_pool3 = tf.add(self.upscore_pool4, score_pool3)

            self.upscore8 = self.deconv_layer(self.fuse_pool3, 18, 18, 10, 5, "upscore8")
            self.pred_up = tf.argmax(self.upscore8, dimension=3)

            self.data_dict = None#For deconv_layer, k_size was calculated by hand, the value has been changed cmpared to the official code

    def avg_pool(self, bottom, filter_size, stride, name):
        return tf.nn.avg_pool(bottom, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def max_pool(self, bottom, filter_size, stride, name):
        return tf.nn.max_pool(bottom, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, filter_size, stride, name, padding="SAME", pad=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
            if pad is not None:
                bottom = tf.pad(bottom, [[0,0],[pad[0],pad[0]],[pad[1],pad[1]],[0,0]],"CONSTANT")
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            print name
            print relu.get_shape()
            return relu

    def deconv_layer(self, bottom, in_channels, out_channels, filter_size, stride, name, padding="VALID", pad=None):
        with tf.variable_scope(name):
            strides = [1, stride, stride,  1]
            in_shape = tf.shape(bottom)
            h = ((in_shape[1] - 1) * stride) + filter_size
            w = ((in_shape[2] - 1) * stride) + filter_size
            output_shape = [in_shape[0], h, w, out_channels]
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
            if pad is not None:
                bottom = tf.pad(bottom, [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], "CONSTANT")
            deconv = tf.nn.conv2d_transpose(bottom, filt, output_shape, strides=strides, padding)
            print name
            print deconv.get_shape()
            return deconv

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            print name
            print fc.get_shape()
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        if self.data_dict is not None and name in self.data_dict:
            init_w = tf.constant(self.data_dict[name][0])
            init_b = tf.constant(self.data_dict[name][1])
            #print self.data_dict[name][0]
            filters = tf.get_variable(name+"_filters", initializer=init_w)
            biases = tf.get_variable(name+"_biases", initializer=init_b)
            self.var_dict[(name, 0)] = filters
            self.var_dict[(name, 1)] = biases
            print("load success")
        else:
            print("Lose model!")
            init_w = tf.truncated_normal_initializer(0.0, 0.001)
            init_b = tf.truncated_normal_initializer(0.0, 0.001)
            filters = tf.get_variable(name+"_filters", [filter_size, filter_size, in_channels, out_channels], initializer=init_w)
            biases = tf.get_variable(name+"_biases", [out_channels], initializer=init_b)
            self.var_dict[(name, 0)] = filters
            self.var_dict[(name, 1)] = biases

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        if self.data_dict is not None and name in self.data_dict:
            init_w = tf.constant(self.data_dict[name][0])
            init_b = tf.constant(self.data_dict[name][1])
            weights = tf.get_variable(name+"_weights", initializer=init_w)
            biases = tf.get_variable(name+"_biases", initializer=init_b)
            self.var_dict[(name, 0)] = weights
            self.var_dict[(name, 1)] = biases
        else:
            init_w = tf.truncated_normal_initializer(0.0, 0.001)
            init_b = tf.truncated_normal_initializer(0.0, 0.001)
            weights = tf.get_variable(name+"_weights", [in_size, out_size], initializer=init_w)
            biases = tf.get_variable(name+"_biases", [out_size], initializer=init_b)
            self.var_dict[(name, 0)] = weights
            self.var_dict[(name, 1)] = biases
        return weights, biases

    def save_npy(self, sess, npy_path=None):
        assert isinstance(sess, tf.InteractiveSession)

        data_dict = {}
        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path