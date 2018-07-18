#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf

class PongConvNet(object):

    def __init__(self, num_frames, num_images, image_size=(299, 299), num_channels=3):
        self.num_frames = num_frames
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.input_width = image_size[0]
        self.input_height = image_size[1]
        self.num_channels = num_channels

    def create_weights(self, shape, stddev=0.05):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    def create_biases(self, shape, value=0.05):
        return tf.Variable(tf.constant(value, shape=shape))

    def create_convolution_layer(self, input, conv_size, num_channels, pooling=True):
        weights = self.create_weights(shape=[conv_size, conv_size, num_channels])
        biases = self.create_biases(shape=[conv_size, conv_size, num_channels])
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 3, 4, 1], padding='SAME')

        layer += biases

        if pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 3, 4, 1],
                                   strides=[1, 3, 4, 1],
                                   padding='SAME')
        layer = tf.nn.relu(layer)

        return layer

    def create_flatten_layer(self, layer):
        shape = layer.get_shape()
        num_features = shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])

        return layer

    def create_fc_layer(self, input, num_inputs, num_outputs):
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)
        layer = tf.matmul(input, weights) + biases

        return layer


    def build_net(self):

        input_x = tf.placeholder(tf.float32, [None, self.input_width, self.input_height, self.num_frames])
        layer_conv1 = self.create_convolution_layer(input=input_x,
                                                    conv_size=32,
                                                    num_channels=self.num_channels)
        layer_conv2 = self.create_convolution_layer(input=layer_conv1,
                                                    conv_size=64,
                                                    num_channels=self.num_channels)
        layer_conv3 = self.create_convolution_layer(input=layer_conv2,
                                                    conv_size=64,
                                                    num_channels=self.num_channels)
        layer_flat = self.create_flatten_layer(layer_conv3)

        layer_fc1 = self.create_fc_layer(layer_flat,
                                         num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                         num_outputs=512)
        layer = tf.nn.relu(layer_fc1)

        layer_fc2 = self.create_fc_layer(layer,
                                         num_inputs=512,
                                         num_outputs=3)
        return input_x, layer_fc2

    def predict(self, input_layer, y_true):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=input_layer,
                                                                labels=y_true)
        cost = tf.reduce_mean(cross_entropy)

        return cost

    def optimize(self, cost):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        batch_size = 16

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)

