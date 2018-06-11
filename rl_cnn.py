#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf

class PongConvNet(object):

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def create_weights(self, shape, stddev=0.05):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    def create_bias(self, shape, value=0.05):
        return tf.Variable(tf.constant(value, shape=shape))

    def create_convolution_layer(self, input, conv_size, num_channels, ):
        weights = self.create_weights(shape=[conv_size, conv_size, num_channels])
        biases = self.create_bias(shape=[3])
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 3, 4, 1], padding='SAME')

        layer += biases

        layer = tf.nn.relu(layer)

        return layer
