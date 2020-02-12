from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pdb
import re
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.callbacks.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from hierarchy_cluster import *
from base_network import *

from keras.models import Sequential
tf.keras.backend.clear_session()  # For easy reset of notebook state.
import keras.backend as K
import keras.losses
import os
from tensorflow.python.ops import math_ops

from tensorflow import keras
from tensorflow.keras import layers

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def mean_squared_error_int(y_true, y_pred):
    return K.mean(K.square((y_pred) - (y_true)), axis=-1)

def custom_mean_abs_loss(y_actual, y_predicted):
    error_sq = K.abs(y_predicted-y_actual)/y_actual
    return K.mean(error_sq)

def custom_mean_abs_exp_loss(y_actual, y_predicted):
    error_sq = K.abs(K.exp(y_predicted)-y_actual)/y_actual
    return K.mean(error_sq)

def custom_mean_abs_log_loss(y_actual, y_predicted):
    error_sq = K.abs(y_predicted-K.log(y_actual))/K.log(y_actual)
    return K.mean(error_sq)

def custom_mean_abs_loss_v3(y_actual, y_predicted):
    error_sq = K.square(y_predicted-K.log(y_actual))
    return K.mean(error_sq, axis=-1)

class SingleOutputNetwork(BaseNetwork):
    def __init__(self, args, parameters_data, cost_data, orig_cost_data, name="network"):
        self.args = args
        self.parameters_data = parameters_data
        self.cost_data = cost_data
        self.orig_cost_data = orig_cost_data
        samples_count = int(args.n)
        unique_elements = len(set(cost_data.reshape(cost_data.size).tolist()))
        self.n_out_fmaps = 1
        #self.nodes = 128
        self.train_count = 0
        self.val_count = 0 
        self.test_count = 0 
        self.nodes = int(args.nodes)
        self.n_in_fmaps = parameters_data.shape[1]
        convs = int(args.convs)
        last_layer_nodes = int(args.last_layer_nodes)
        self.name = name
        self.validation_split = float(args.validation_split)
        self.transfer_learning = args.transfer_learning
        self.step = -1
        hidden_layers = []
        if convs==0:
            hidden_layers = [(self.nodes, 'tanh', 'dense1', {"dropout":0.4}),
                         (self.nodes, 'tanh', 'dense2', {"dropout":0.4}),
                         (self.nodes, 'tanh', 'dense3', {"dropout":0.4}),
                         (last_layer_nodes,  'relu', 'dense4', {}),
                         (self.n_out_fmaps, 'relu', 'predict', {})]
            self.model = BaseNetwork.get_uncompiled_model(self, self.n_in_fmaps, hidden_layers, name)
        elif convs==1:
            hidden_layers = [(1, None, 'reshape1', {'type': 'Reshape', 'shape':(self.n_in_fmaps, 1)}),
                         (32, 'tanh', 'conv1',  {'type': 'Conv1D', 'kernel_size':3, "dropout":0.4}),
                         (self.nodes, None, 'reshape2', {'type': 'Reshape', 'shape':(1, 32*(self.n_in_fmaps-2))}),
                         (self.nodes, 'tanh', 'dense2', {"dropout":0.4}),
                         (self.nodes, 'tanh', 'dense3', {"dropout":0.4}),
                         (last_layer_nodes,  'relu', 'dense4', {}),
                         (self.n_out_fmaps, 'relu', 'predict', {})]
            self.model = BaseNetwork.get_uncompiled_model(self, self.n_in_fmaps, hidden_layers, name)
        elif convs==2:
            hidden_layers = [(1, None, 'reshape1', {'type': 'Reshape', 'shape':(self.n_in_fmaps, 1)}),
                         (32, 'tanh', 'conv1',  {'type': 'Conv1D', 'kernel_size':3, "dropout":0.4}),
                         (64, 'tanh', 'conv2',  {'type': 'Conv1D', 'kernel_size':3, "dropout":0.4}),
                         (self.nodes, None, 'reshape2', {'type': 'Reshape', 'shape':(1, 64*(self.n_in_fmaps-2-2))}),
                         (self.nodes, 'tanh', 'dense3', {"dropout":0.4}),
                         (last_layer_nodes,  'relu', 'dense4', {}),
                         (self.n_out_fmaps, 'relu', 'predict', {})]
            self.model = BaseNetwork.get_uncompiled_model(self, self.n_in_fmaps, hidden_layers, name)
        elif convs==3:
            hidden_layers = [(1, None, 'reshape1', {'type': 'Reshape', 'shape':(self.n_in_fmaps, 1)}),
                         (32, 'tanh', 'conv1',  {'type': 'Conv1D', 'kernel_size':3, "dropout":0.4}),
                         (64, 'tanh', 'conv2',  {'type': 'Conv1D', 'kernel_size':3, "dropout":0.4}),
                         (128, 'tanh', 'conv3',  {'type': 'Conv1D', 'kernel_size':3, "dropout":0.4}),
                         (self.nodes, None, 'reshape2', {'type': 'Reshape', 'shape':(1, 128*(self.n_in_fmaps-2-2-2))}),
                         (last_layer_nodes,  'relu', 'dense4', {}),
                         (self.n_out_fmaps, 'relu', 'predict', {})]
            self.model = BaseNetwork.get_uncompiled_model(self, self.n_in_fmaps, hidden_layers, name)
        elif convs==4:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense1")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==5:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(256, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==6:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(128, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==7:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(64, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==8:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(512, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==9:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(512, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==10:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(128, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==11:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(256, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==12:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', name="conv1")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn0")(x)
            x = layers.Activation('tanh', name="tanh0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(32, kernel_size=3, strides=1,  padding='same', name="conv2")(x)
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn1")(x)
            x = layers.Activation('tanh', name="tanh1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 256), name="reshape1")(x)
            x = layers.Dense(64, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==13:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, activation='tanh', kernel_size=5, strides=1, padding='same', name="conv1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(64, activation='tanh', kernel_size=5, strides=1, name="conv2")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 64*(self.n_in_fmaps-4)), name="reshape1")(x)
            x = layers.Dense(self.nodes, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==14:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, activation='tanh', kernel_size=3, strides=1, name="conv1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(64, activation='tanh', kernel_size=3, strides=1, name="conv2")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 64*(self.n_in_fmaps-4)), name="reshape1")(x)
            x = layers.Dense(self.nodes, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        elif convs==15:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, activation='tanh', kernel_size=3, strides=1, name="conv1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(64, activation='tanh', kernel_size=3, strides=1, name="conv2")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 64*(self.n_in_fmaps-4)), name="reshape1")(x)
            x = layers.Dense(self.nodes, activation="tanh", use_bias=True, name="dense0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(last_layer_nodes, activation="tanh", use_bias=True, name="dense1")(x)
            x = layers.Dense(1, activation="relu", use_bias=True, name="dense2")(x)
            x = layers.Cast(dtype='int32')(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
            self.model_name = "model.png"
            if name != "":
                self.model_name = name+"_model.png"
            tf.keras.utils.plot_model(self.model, to_file=self.model_name)
            self.model.summary()
        self.loss_function = custom_mean_abs_exp_loss
        if args.loss == 'custom_mean_abs_loss':
            self.loss_function = custom_mean_abs_loss
        elif args.loss == 'mean_squared_error':
            self.loss_function = mean_squared_error 
        elif args.loss == 'mean_squared_error_int':
            self.loss_function = mean_squared_error_int
        elif args.loss == 'custom_mean_abs_log_loss':
            self.loss_function = custom_mean_abs_log_loss
        keras.losses.custom_loss = self.loss_function
        self.model = BaseNetwork.get_compiled_model(self, self.model, loss=self.loss_function, optimizer='adam', metrics=["mse"])
        self.batch_size = min(self.parameters_data.shape[0], int(args.batch_size))
        self.epochs = int(args.epochs)
        self.disable_icp = False

    def GetEpochs(self):
        return self.epochs

    def GetBatchSize(self):
        return self.batch_size

    def SetTransferLearning(self, count):
        self.transfer_learning = count

    def DisableICP(self, flag=True):
        self.disable_icp = flag

    def preprocess_data_incremental(self, step, train_idx, val_idx, test_idx):
        training_idx = np.concatenate((train_idx, val_idx)) 
        np.save("step{}-train-indices.npy".format(step), training_idx)
        np.save("step{}-test-indices.npy".format(step), test_idx)
        val_count = len(training_idx) * self.validation_split
        train_count = len(training_idx) - val_count
        test_count = len(test_idx)
        self.train_count = train_count
        self.val_count = val_count
        self.test_count = test_count
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        self.step = step
        self.validation_split = val_count / (train_count+val_count)
        print("Total count:"+str(train_count+val_count+test_count))
        print("Train count:"+str(train_count))
        print("Val count:"+str(val_count))
        print("Test count:"+str(test_count))
        print("Train-Validation Split:"+str(self.validation_split))
        #print("Tr_indices:"+str(training_idx))
        #print("test_indices:"+str(test_idx))
        x_train = parameters[training_idx,:].astype('float')
        y_train = cost_data[training_idx,:].astype('float')
        z_train = orig_cost_data[training_idx,:].astype('float') 
        x_test = parameters[test_idx,:].astype('float')
        y_test = cost_data[test_idx,:].astype('float')
        z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def preprocess_data(self, cluster_data=False):
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        train_count = int(parameters.shape[0]*float(self.args.train_test_split))
        test_count = parameters.shape[0] - train_count
        print("Total count:"+str(parameters.shape[0]))
        print("Train count:"+str(train_count))
        print("Test count:"+str(test_count))
        self.train_count = train_count
        self.val_count = int(train_count * self.validation_split)
        self.test_count = test_count
        indices = np.random.permutation(parameters.shape[0])
        training_idx = indices[:train_count]
        test_idx = indices[train_count:]
        np.save("train-indices.npy", training_idx)
        np.save("test-indices.npy", test_idx)
        if cluster_data:
            tr_indices = []
            test_indices = []
            self.classification_bins = 1000
            cost_groups = [[] for i in range(self.classification_bins)]
            max_entries = 0
            self.cost_bin, self.labels = self.GetHierarchicalDistributionBins(self.classification_bins)
            cost_labels = self.labels
            for index, label in enumerate(cost_labels):
                cost_groups[label[0]].append(index)
                max_entries = max(max_entries, len(cost_groups[label[0]]))
            for index in range(max_entries):
                for cgroup in cost_groups:
                    if len(cgroup) > 0 and len(tr_indices) < train_count:
                        tr_indices.append(cgroup.pop())
                    if len(cgroup) > 0 and len(test_indices) < test_count:
                        test_indices.append(cgroup.pop())
            training_norand_idx = np.array(tr_indices).astype('int')
            test_norand_idx = np.array(test_indices).astype('int')
            train_index_indices = np.random.permutation(training_norand_idx.shape[0])
            test_index_indices = np.random.permutation(test_norand_idx.shape[0])
            training_idx = training_norand_idx[train_index_indices,]
            test_idx = test_norand_idx[test_index_indices,]
        print("Tr_indices:"+str(training_idx))
        print("test_indices:"+str(test_idx))
        print("Tr_indices count:"+str(training_idx.size))
        print("test_indices count:"+str(test_idx.size))
        x_train = parameters[training_idx,:].astype('float')
        y_train = cost_data[training_idx,:].astype('float')
        z_train = orig_cost_data[training_idx,:].astype('float') 
        x_test = parameters[test_idx,:].astype('float')
        y_test = cost_data[test_idx,:].astype('float')
        z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def load_train_test_data(self, step=-1):
        if step == -1:
            training_idx = np.load("train-indices.npy")
            test_idx     = np.load("test-indices.npy")
        else:
            training_idx = np.load("step{}-train-indices.npy".format(step))
            test_idx     = np.load("step{}-test-indices.npy".format(step))
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        x_train = parameters[training_idx,:].astype('float')
        y_train = cost_data[training_idx,:].astype('float')
        z_train = orig_cost_data[training_idx,:].astype('float') 
        x_test = parameters[test_idx,:].astype('float')
        y_test = cost_data[test_idx,:].astype('float')
        z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test
        self.train_count = len(training_idx) * (1-self.validation_split)
        self.val_count = len(training_idx) * self.validation_split
        self.test_count = len(test_idx)

    def evaluate(self, x_test, y_test, z_test, tags=""):
        if x_test.size == 0:
            return
        print("***********************************************")
        print('\n# Generate predictions for 3 samples of '+tags)
        predictions = self.model.predict(x_test, batch_size=self.GetBatchSize())
        print(predictions[0], np.log(y_test[0]), np.exp(predictions[0]), y_test[0], z_test[0])
        print(predictions[1], np.log(y_test[1]), np.exp(predictions[1]), y_test[1], z_test[1])
        print(predictions[2], np.log(y_test[2]), np.exp(predictions[2]), y_test[2], z_test[2])
        print('predictions shape:', predictions.shape)
        estimation_error = []
        recalc_error = []
        mean_square = False
        for sample_idx in range(len(y_test)):
            nn_out = np.exp(predictions[sample_idx])
            # recalc_val = np.exp(y_test[sample_idx])
            sim_val = y_test[sample_idx]

            cycle_diff = np.float(np.abs(sim_val-nn_out))
            cycle_error = cycle_diff/sim_val
            if mean_square:
                cycle_error = cycle_diff * cycle_diff
            estimation_error.append(cycle_error)

            # recalc_error_val = np.float(np.abs(sim_val-recalc_val))/sim_val
            # recalc_error.append(recalc_error)
        print(tags+" Mean: ", np.mean(estimation_error), "Max: ", np.max(estimation_error), "Min: ", np.min(estimation_error))
        #print(tags+" MeanAct: ", np.mean(estimation_act_error), "MaxAct: ", np.max(estimation_act_error), "MinAct: ", np.min(estimation_act_error))

    def GetHierarchicalDistributionBins(self, classification_bins):
        cost_data = self.cost_data
        print("Cost shape:"+str(cost_data.shape))
        min_cost = np.min(cost_data)
        max_cost = np.max(cost_data)
        cost_data_rs = cost_data.reshape(cost_data.size)
        index_list = [index for index in range(len(cost_data_rs.tolist()))]
        cost_list = cost_data_rs.tolist()
        if True:
            plt.scatter(index_list, cost_list,c='black')
            plt.xlabel('Index')
            plt.ylabel('Cycles')
            plt.savefig("cycles_sorted.png")
            plt.close()
        hierarchical = Hierarchical(n_clusters = classification_bins)
        hierarchical.fit(cost_data.tolist())
        colors = mcolors.CSS4_COLORS
        colors_key = colors.keys()
        if True:
            plt.scatter(index_list, cost_list, c=hierarchical.labels_.astype(float))
            plt.savefig("cluster.png")
            plt.close()
        cost_bin = hierarchical.cluster_centers_
        hierarchical.labels_.astype(int)
        return cost_bin, hierarchical.labels_.reshape((hierarchical.labels_.size,1))

    def load_model(self, model_name):
        print("Loading the checkpoint: "+model_name)
        keras.losses.custom_loss = self.loss_function
        self.model.load_weights(model_name)

    def run_model(self):
        class TestCallback(Callback):
            def __init__(self, test_data, fh):
                self.test_data = test_data
                self.fh = fh

            def on_epoch_end(self, epoch, logs={}):
                x, y = self.test_data
                loss, acc = self.model.evaluate(x, y, verbose=0)
                fh.write(str(epoch)+", "+str(loss)+", "+str(acc))
                #print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        x_train, y_train, z_train = self.x_train, self.y_train, self.z_train
        x_test, y_test, z_test    = self.x_test, self.y_test, self.z_test   
        # Train the model by slicing the data into "batches"
        # of size "batch_size", and repeatedly iterating over
        # the entire dataset for a given number of "epochs"
        if self.args.no_run:
            return
        if self.args.evaluate and self.args.icp != "":
            print("Loading checkpoint file:"+self.args.icp)
            keras.losses.custom_loss = self.loss_function
            self.model.load_weights(self.args.icp)
            print('\n# Evaluate on test data')
            self.evaluate(x_train, y_train, z_train, "training")
            self.evaluate(x_test, y_test, z_test, "test")
            loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
            print("Train Loss: "+str(loss))
            loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            print("Test Loss: "+str(loss))
        elif self.args.icp != "" and not self.disable_icp:
            print("Loading checkpoint file:"+self.args.icp)
            keras.losses.custom_loss = self.loss_function
            self.model.load_weights(self.args.icp)
            #self.model = keras.models.load_model(self.args.icp)
            #print('\n# Evaluate on test data')
            #self.evaluate(x_train, y_train, z_train, "training")
            #self.evaluate(x_test, y_test, z_test, "test")
        if not self.args.evaluate:
            pretag = "weights-improvement-"
            if self.step != -1:
                pretag = "step"+str(self.step)+"-train"+str(self.train_count)+"-val"+str(self.val_count)+"-"+pretag
            filepath = pretag + "{epoch:02d}-loss{loss:.4f}-valloss{val_loss:.4f}.hdf5"
            model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
            callbacks_list = [model_checkpoint]
            #if self.args.incremental_learning:
                #fh = open("step"+str(self.step)+"-train"+str(self.train_count)+"-val"+str(self.val_count)+"testloss.csv", "w")
                #test_checkpoint = TestCallback((x_test, y_test), fh)
                #callbacks_list.append(test_checkpoint)
            print('# Fit model on training data')
            with tf.device('/cpu:0'):
                if self.transfer_learning != "-1":
                    for layer in self.model.layers[:int(self.args.transfer_learning)]:
                        print("Frozen Layer: "+layer.name)
                        layer.trainable = False
                history = self.model.fit(x_train, y_train,
                                batch_size=self.GetBatchSize(),
                                epochs=self.GetEpochs(),
                                # We pass some validation for
                                # monitoring validation loss and metrics
                                # at the end of each epoch
                                #validation_data=(x_test, y_test),
                                validation_split = self.validation_split,
                                callbacks=callbacks_list,
                                shuffle=True
                                )
                print("Completed model fitting")
            #fh.close()
            if not self.args.incremental_learning:
                print("Evaluating training and test accuracies")
                self.evaluate(x_train, y_train, z_train, "training")
                self.evaluate(x_test, y_test, z_test, "test")
                self.model.save_weights('model.hdf5')
                print(history.history.keys())

            # Plot training & validation loss values
            plot_loss = self.args.plot_loss
            if plot_loss:
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                #plt.yscale('log')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.savefig("loss.png")
                plt.close()

            # The returned "history" object holds a record
            # of the loss values and metric values during training
            #print('\nhistory dict:', history.history)

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`

    def get_last_cp_model(self, all_files):
        epoch_re = re.compile(r'weights-improvement-([0-9]+)-')
        max_epoch = 0
        last_icp = ''
        for index, icp_file in enumerate(all_files):
            epoch_flag = epoch_re.search(icp_file)
            epoch=0 #loss0.4787-valloss0.4075.hdf5a
            if epoch_flag:
                epoch = int(epoch_flag[1]) 
            if epoch > max_epoch:
                max_epoch = epoch
                last_icp = icp_file
        return last_icp

    def evaluate_model(self, all_files, outfile="test-output.csv"):
        epoch_re = re.compile(r'weights-improvement-([0-9]+)-')
        loss_re = re.compile(r'-loss([^-]*)-')
        valloss_re = re.compile(r'-valloss(.*)\.hdf5')
        #step2-train350-val350-weights-improvement-610-loss0.1988-valloss0.1341.hdf5
        step_re = re.compile(r'step([^-]*)-')
        traincount_re = re.compile(r'-train([^-]*)-')
        valcount_re = re.compile(r'val([0-9][^-]*)-')
        with open(outfile, "w") as fh:
            hdrs = [ "Epoch", "TrainLoss", "ValLoss", "TestLoss" ]
            print("Calculating test accuracies "+str(len(all_files)))
            for index, icp_file in enumerate(all_files):
                epoch_flag = epoch_re.search(icp_file)
                loss_flag = loss_re.search(icp_file)
                valloss_flag = valloss_re.search(icp_file)
                step_flag = step_re.search(icp_file)
                traincount_flag = traincount_re.search(icp_file)
                valcount_flag = valcount_re.search(icp_file)
                epoch=0 #loss0.4787-valloss0.4075.hdf5a
                train_loss = 0.0
                val_loss = 0.0
                traincount = 0
                valcount = 0
                step = -1 
                if epoch_flag:
                    epoch = int(epoch_flag[1]) 
                if loss_flag:
                    train_loss = float(loss_flag[1]) 
                if valloss_flag:
                    val_loss = float(valloss_flag[1]) 
                if step_flag:
                    step = int(step_flag[1])
                if traincount_flag:
                    traincount = int(float(traincount_flag[1]))
                if valcount_flag:
                    valcount = int(float(valcount_flag[1]))
                self.model.load_weights(icp_file)
                if self.args.load_train_test:
                    self.load_train_test_data(step)
                x_train, y_train, z_train = self.x_train, self.y_train, self.z_train
                x_test, y_test, z_test    = self.x_test, self.y_test, self.z_test   
                keras.losses.custom_loss = self.loss_function
                loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                if fh != None:
                    if index == 0:
                        if step_flag:
                            hdrs.append("Step")
                            hdrs.append("TrainCount")
                            hdrs.append("ValCount")
                        fh.write(", ".join(hdrs)+"\n")
                    data = [str(epoch), str(train_loss), str(val_loss), str(loss)]
                    if step_flag:
                        data.extend([str(step), str(traincount), str(valcount)])
                    fh.write(", ".join(data)+"\n")
                #print('Testing epoch:{} train_loss: {}, val_loss: {}, test_loss: {}\n'.format(epoch, train_loss, val_loss, loss))
            fh.close()
