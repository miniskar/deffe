from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pdb
import re
import shlex
import argparse
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.callbacks.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import matplotlib.colors as mcolors
import glob

from keras.models import Sequential
tf.keras.backend.clear_session()  # For easy reset of notebook state.
import keras.backend as K
import keras.losses
import os
from tensorflow.python.ops import math_ops

from tensorflow import keras
from tensorflow.keras import layers
from baseml import *

checkpoint_dir = "checkpoints"
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

class KerasCNN(BaseMLModel):
    def __init__(self, framework):
        self.framework = framework
        self.args = self.ParseArguments()
        self.step = -1
        self.step_start = framework.args.step_start
        self.step_end= framework.args.step_end
        self.epochs = int(self.args.epochs)
        if framework.args.epochs != '-1':
            self.epochs = int(framework.args.epochs)
        self.nodes = int(self.args.nodes)
        self.convs = int(self.args.convs)
        self.validation_split = float(self.args.validation_split)
        self.tl_freeze_layers = self.args.tl_freeze_layers
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def ParseArguments(self):
        arg_string = self.framework.config.GetModel().arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-train-test-split', dest='train_test_split', default="1.00")
        parser.add_argument('-validation-split', dest='validation_split', default="0.20")
        parser.add_argument('-icp', dest='icp', default="")
        parser.add_argument('-epochs', dest='epochs', default="50")
        parser.add_argument('-batch-size', dest='batch_size', default="256")
        parser.add_argument('-tl_freeze_layers', dest='tl_freeze_layers', default="-1")
        parser.add_argument('-convs', dest='convs', default="2")
        parser.add_argument('-tl-samples', dest='tl_samples', action='store_true')
        parser.add_argument('-no-run', dest='no_run', action='store_true')
        parser.add_argument('-evaluate-only', dest='evaluate', action='store_true')
        parser.add_argument('-load-train-test', dest='load_train_test', action='store_true')
        parser.add_argument('-plot-loss', dest='plot_loss', action='store_true')
        parser.add_argument('-loss', dest='loss', default='')
        parser.add_argument('-nodes', dest='nodes', default="256")
        parser.add_argument('-last-layer-nodes', dest='last_layer_nodes', default="32")
        args = parser.parse_args(shlex.split(arg_string))
        return args

    def Initialize(self, step, headers, parameters_data, cost_data, name="network"):
        args = self.args
        self.step = step
        print("Headers: "+str(headers))
        self.parameters_data = parameters_data
        self.cost_data = cost_data
        self.orig_cost_data = cost_data 
        self.n_out_fmaps = 1
        #self.nodes = 128
        self.train_count = 0
        self.val_count = 0 
        self.test_count = 0 
        self.n_in_fmaps = parameters_data.shape[1]
        last_layer_nodes = int(args.last_layer_nodes)
        self.name = name
        if self.convs==0:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Dense(self.nodes, activation="tanh", name="dense0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(self.nodes, activation="tanh", name="dense1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(self.nodes, activation="tanh", name="dense2")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(last_layer_nodes, activation="relu", name="dense3")(x)
            x = layers.Dense(self.n_out_fmaps, activation="relu", name="predict")(x)
            self.model = keras.Model(inputs=inputs, outputs=x)
        elif self.convs==2:
            inputs = keras.Input(shape=(self.n_in_fmaps,), name='parameters')
            x = inputs
            x = layers.Reshape((self.n_in_fmaps, 1), name="reshape0")(x)
            x = layers.Conv1D(32, activation='tanh', kernel_size=3, strides=1, name="conv1")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Conv1D(64, activation='tanh', kernel_size=3, strides=1, name="conv2")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Reshape((1, 64*(self.n_in_fmaps-4)), name="reshape1")(x)
            x = layers.Dense(self.nodes, activation="tanh", name="dense0")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Dense(last_layer_nodes, activation="relu", name="dense1")(x)
            x = layers.Dense(1, activation="relu", name="dense2")(x)
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
        self.model = self.get_compiled_model(self.model, loss=self.loss_function, optimizer='adam', metrics=["mse"])
        batch_size = int(args.batch_size)
        if self.framework.args.batch_size != '-1':
            batch_size = int(self.framework.args.batch_size)
        self.batch_size = min(self.parameters_data.shape[0], batch_size)
        self.disable_icp = False

    def get_compiled_model(self, model, loss='categorical_crossentropy', optimizer = keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy']):
        model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
        for l in model.layers:
            print(l.input_shape, l.output_shape)
        return model

    def GetEpochs(self):
        return self.epochs

    def GetBatchSize(self):
        return self.batch_size

    def SetTransferLearning(self, count):
        self.tl_freeze_layers = count

    def DisableICP(self, flag=True):
        self.disable_icp = flag

    def preprocess_data(self):
        BaseMLModel.preprocess_data(self, self.parameters_data, self.cost_data, self.orig_cost_data, self.args.train_test_split, self.validation_split)
        if self.args.tl_samples and self.step != self.step_start:
            all_files = glob.glob(os.path.join(checkpoint_dir, "step{}-*.hdf5".format(self.step-1)))
            last_cp = self.get_last_cp_model(all_files)
            if last_cp != '':
                self.load_model(last_cp)

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

    def Inference(self):
        return None

    def load_model(self, model_name):
        print("Loading the checkpoint: "+model_name)
        keras.losses.custom_loss = self.loss_function
        self.model.load_weights(model_name)

    def Train(self):
        pdb.set_trace()
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
            model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
            callbacks_list = [model_checkpoint]
            print('# Fit model on training data')
            if self.tl_freeze_layers != "-1":
                for layer in self.model.layers[:int(self.args.tl_freeze_layers)]:
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
        return (0.0, 0.0)

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


def GetObject(framework):
    obj = KerasCNN(framework)
    return obj

