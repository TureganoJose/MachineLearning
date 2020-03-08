
# https://www.researchgate.net/publication/301648615_End_to_End_Learning_for_Self-Driving_Cars

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D, Input
from tensorflow.keras import regularizers
from tensorflow.keras import Model

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def cnn_layers(model_input):
    # Normalise
    model_input /= 255.0
    x = keras.layers.Lambda(lambda x_input: x_input/255.0)(model_input)

    # Dilation layer 1
    x = keras.layers.Conv2D(3, (5, 5), strides=(2, 2), padding='same', data_format='channels_last',
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                            bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = keras.layers.Conv2D(24, (5, 5), strides=(2, 2), padding='valid', data_format=None,
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                            bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
    # Dilation layer 2
    x = keras.layers.Conv2D(36 (5, 5), strides=(2, 2), padding='same', data_format=None,
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                            bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                            bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
    # Dilation layer 3
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                            bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
    # Fully connected
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    prediction = Dense(1, activation='softmax')(x)
    return prediction


input_depth = 3
input_height = 66
input_width = 200
num_channels = 3

kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 20

# Float64 by default in layers
tf.keras.backend.set_floatx('float64')

model_input = Input(shape=(input_depth, input_height, input_width))
model_output = cnn_layers(model_input)
train_model = Model(inputs=model_input, outputs=model_output)

train_model.compile(loss='mean_squared_error',
                    optimizer=keras.optimizers.Nadam(lr=learning_rate),
                    metrics=['accuracy'])