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
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D
from tensorflow.keras import Model

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

segments = np.load('segments.npy')
labels = np.load('labels.npy')

reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

Train_Batch_size = train_x.shape
batch_size = Train_Batch_size[0]

# Checking data
#total = 0
#for iPlot in range(batch_size):
#    x_data1 = train_x[iPlot, 0, :, 0]
#    x_data2 = train_x[iPlot, 0, :, 1]
#    x_data3 = train_x[iPlot, 0, :, 2]
#    for iAct in range(6):
#        total = total + train_y[iPlot, iAct]
#        if train_y[iPlot, iAct] == 1:
#            y_data1 = [iAct]*90
#    print(total)
#    total = 0
#
#    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(15, 10), sharex=True)
#    plot_axis(ax0, range(90), x_data1, 'x-axis')
#    plot_axis(ax1, range(90), x_data2, 'y-axis')
#    plot_axis(ax2, range(90), x_data3, 'z-axis')
#    plot_axis(ax3, range(90), y_data1, 'Activity')
#    plt.subplots_adjust(hspace=0.2)
#    #fig.suptitle(activity)
#    plt.subplots_adjust(top=0.90)
#    plt.show()


input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.001
training_epochs = 100

# Float64 by default in layers
tf.keras.backend.set_floatx('float64')


# Add a channels dimension
#train_x = train_x[..., tf.newaxis]
#test_x = test_x[..., tf.newaxis]
#
#train_ds = tf.data.Dataset.from_tensor_slices(
#    (train_x, train_y)).shuffle(17079).batch(batch_size)
#
#test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)

# Defining model
#class MyModel(Model):
#    def __init__(self,
#                 loss_object,
#                 optimizer,
#                 train_loss,
#                 train_metric,
#                 test_loss,
#                 test_metric):
#        '''
#            Setting all the variables for our model.
#        '''
#        super(MyModel, self).__init__()
#
#        self.conv1 = DepthwiseConv2D((60, 60), 3, activation='relu')
#        self.conv1 = DepthwiseConv2D(kernel_size, activation='relu', strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None,
#                                     dilation_rate=(1, 1), activation=None, use_bias=True,
#                                     depthwise_initializer='glorot_uniform', bias_initializer='zeros',
#                                     depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#                                     depthwise_constraint=None, bias_constraint=None)
#
#        self.flatten = Flatten()
#        self.d1 = Dense(128, activation='relu')
#        self.d2 = Dense(10, activation='softmax')
#
#        self.loss_object = loss_object
#        self.optimizer = optimizer
#        self.train_loss = train_loss
#        self.train_metric = train_metric
#        self.test_loss = test_loss
#        self.test_metric = test_metric

model = keras.models.Sequential()
model.add( DepthwiseConv2D((1, 60), activation='relu', strides=(1, 1), padding='valid', depth_multiplier=1,
                             data_format='channels_last',
                             dilation_rate=(1, 1),  use_bias=True,
                             depthwise_initializer='glorot_uniform', bias_initializer='zeros',
                             depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             depthwise_constraint=None, bias_constraint=None,batch_input_shape=(batch_size,1,90,3)))
model.add( MaxPool2D(pool_size=(1,20),strides=2,padding='valid'))
model.add( DepthwiseConv2D((1, 6), activation='relu', strides=(1, 1), padding='valid', depth_multiplier=1,
                             data_format='channels_last',
                             dilation_rate=(1, 1), use_bias=True,
                             depthwise_initializer='glorot_uniform', bias_initializer='zeros',
                             depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             depthwise_constraint=None, bias_constraint=None))
model.add(Flatten())
model.add(Dense(1000, activation='tanh'))
model.add( Dense(6, activation='softmax'))

model.summary()
model.layers

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Nadam(lr=learning_rate),
              metrics=['accuracy'])

history = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=training_epochs,
          verbose=1,
          validation_data=(test_x, test_y))

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plt.plot(range(1,training_epochs+1), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
