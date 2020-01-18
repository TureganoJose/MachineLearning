# Implementation of a couple of papers:
# https://arxiv.org/pdf/1905.01697.pdf
# http://aqibsaeed.github.io/2016-11-04-human-activity-recognition-cnn/
#
# Data here:
# http://www.cis.fordham.edu/wisdm/dataset.php

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

segments = np.load('segments.npy')
labels = np.load('labels.npy')

reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 5

total_batchs = train_x.shape[0] // batch_size


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    hola = tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))
    return hola


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


# X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
# Y = tf.placeholder(tf.float32, shape=[None,num_labels])

def loss_function(prediction, labels):
    return -tf.reduce_sum(labels * tf.log(prediction))


def model(model_input):
    # Forward predictions
    c = apply_depthwise_conv(model_input, kernel_size, num_channels, depth)
    p = apply_max_pool(c, 20, 2)
    c = apply_depthwise_conv(p, 6, depth * num_channels, depth // 10)

    shape = c.get_shape().as_list()
    c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

    f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth // 10), num_hidden])
    f_biases_l1 = bias_variable([num_hidden])
    f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
    return y_


def train_step(cnn_model, inputs, targets):
    #with tf.GradientTape() as tape:
    current_loss = loss_function(cnn_model(inputs), targets)
    optimizer.minimize(current_loss)
    print(tf.reduce_mean(current_loss))


# Setup a stochastic gradient descent optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for epoch in range(training_epochs):
    cost_history = np.empty(shape=[1], dtype=float)
    for b in range(total_batchs):
        offset = (b * batch_size) % (train_y.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :, :]
        batch_y = train_y[offset:(offset + batch_size), :]

        train_step(model, batch_x, batch_y)

    updated_loss = current_loss = loss_function(model(test_x), test_y)
    my_string = "epoch {} with updated loss of {}"

    print(my_string.format("epoch", "updated_loss"))
