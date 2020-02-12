from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pdb
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import matplotlib.colors as mcolors
import keras.backend as K

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow import keras
from tensorflow.keras import layers
#from numpy.random import seed 
#seed(2) 
#tf.compat.v1.set_random_seed(2)

class BaseNetwork:
    def __init(self):
        None

    def get_uncompiled_model(self, n_in_fmaps, hidden_layers, name=""):
        inputs = keras.Input(shape=(n_in_fmaps,), name='parameters')
        x = inputs
        for (nodes, act, name, params)  in hidden_layers:
            layer_type = 'Dense'
            if 'type' in params:
                layer_type = params['type']
            if act != None:
                if layer_type == 'Conv1D':
                    x = layers.Conv1D(nodes, kernel_size=params['kernel_size'], activation=act, name=name)(x)
                elif layer_type == 'Reshape':
                    x = layers.Reshape(params['shape'], name=name)(x)
                else:
                    x = layers.Dense(nodes, activation=act, name=name)(x)
            else:
                if layer_type == 'Conv1D':
                    if "padding" in params:
                        x = layers.Conv1D(nodes, kernel_size=params['kernel_size'], padding=params['padding'], name=name)(x)
                    else:
                        x = layers.Conv1D(nodes, kernel_size=params['kernel_size'], name=name)(x)
                elif layer_type == 'Reshape':
                    x = layers.Reshape(params['shape'], name=name)(x)
                else:
                    x = layers.Dense(nodes, name=name)(x)
            if "dropout" in params:
                x = layers.Dropout(rate=params["dropout"])(x)
        model = keras.Model(inputs=inputs, outputs=x)
        self.model_name = "model.png"
        if name != "":
            self.model_name = name+"_model.png"
        tf.keras.utils.plot_model(model, to_file=self.model_name)
        model.summary()
        return model

    #loss='mean_absolute_error',
    #optimizer = 'adam'
    def get_compiled_model(self, model, loss='categorical_crossentropy', optimizer = keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy']):
        model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
        for l in model.layers:
            print(l.input_shape, l.output_shape)
        return model


