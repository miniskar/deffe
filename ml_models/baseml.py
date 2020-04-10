## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
# @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
#         miniskarnr@ornl.gov
#
# Modification:
#              Baseline code
# Date:        Apr, 2020
# **************************************************************************
###
import numpy as np
import re
import pdb
import pandas as pd

checkpoint_dir = "checkpoints"


class BaseMLModel:
    def __init__(self):
        None

    def Initialize(self, headers, parameters, cost_data, train_indexes, val_indexes):
        print("Headers: " + str(headers))
        orig_cost_data = cost_data
        self.headers = headers
        self.parameters_data = parameters
        self.cost_data = cost_data
        self.orig_cost_data = orig_cost_data
        self.train_actual_indexes = train_indexes
        self.val_actual_indexes = val_indexes

    def PreLoadData(self, step, train_test_split, validation_split, shuffle=False):
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        train_count = int(parameters.shape[0] * train_test_split)
        test_count = parameters.shape[0] - train_count
        print("Init Total count:" + str(parameters.shape[0]))
        print("Init Train count:" + str(train_count))
        print("Init Test count:" + str(test_count))
        self.train_count = train_count
        self.val_count = int(train_count * validation_split)
        self.test_count = test_count
        indices = range(parameters.shape[0])
        if shuffle:
            indices = np.random.permutation(parameters.shape[0])
        training_idx = indices[:train_count]
        test_idx = indices[train_count:]
        # print("Tr_indices:"+str(training_idx))
        # print("test_indices:"+str(test_idx))
        # print("Tr_indices count:"+str(training_idx.size))
        # print("test_indices count:"+str(test_idx.size))
        self.training_idx = training_idx
        self.test_idx = test_idx
        x_train = parameters[training_idx,:].astype('float')
        x_test = parameters[test_idx,:].astype('float')
        y_train = np.array([])
        z_train = np.array([])
        y_test = np.array([])
        z_test = np.array([])
        if cost_data.size != 0:
            y_train = cost_data[training_idx,:].astype('float')
            z_train = orig_cost_data[training_idx,:].astype('float') 
            y_test = cost_data[test_idx,:].astype('float')
            z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def SaveTrainValTestData(self, step=-1):
        if step == -1:
            print("Saving train indices: train-indices.npy")
            np.save("train-indices.npy", self.train_actual_indexes)
            print("Saving val indices: val-indices.npy")
            np.save("val-indices.npy", self.val_actual_indexes)
        else:
            print("Saving train indices: step{}-train-indices.npy".format(step))
            np.save("step{}-train-indices.npy".format(step), self.train_actual_indexes)
            print("Saving val indices: step{}-val-indices.npy".format(step))
            np.save("step{}-val-indices.npy".format(step), self.val_actual_indexes)

    def LoadTrainValTestData(self, step=-1):
        if step == -1:
            print("Loading train indices: train-indices.npy")
            train_actual_indexes = np.load("train-indices.npy")
            print("Loading val indices: val-indices.npy")
            val_actual_indexes = np.load("val-indices.npy")
        else:
            print("Loading train indices: step{}-train-indices.npy".format(step))
            train_actual_indexes = np.load("step{}-train-indices.npy".format(step))
            print("Loading val indices: step{}-val-indices.npy".format(step))
            val_actual_indexes = np.load("step{}-val-indices.npy".format(step))
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        train_val_indexes = np.concatenate(
            (self.train_actual_indexes, self.val_actual_indexes), axis=None
        )
        index_hash = {
            target_index: index for index, target_index in enumerate(train_val_indexes)
        }
        train_val_indexes = np.concatenate(
            (train_actual_indexes, val_actual_indexes), axis=None
        )
        training_idx = [index_hash[index] for index in train_val_indexes]
        x_train = parameters[training_idx, :].astype("float")
        y_train = cost_data[training_idx, :].astype("float")
        z_train = orig_cost_data[training_idx, :].astype("float")
        # Get all remaining data other than traininga
        all_indexes = range(parameters.shape[0])
        test_idx = np.delete(all_indexes, training_idx)
        x_test = parameters[test_idx, :].astype("float")
        y_test = cost_data[test_idx, :].astype("float")
        z_test = orig_cost_data[test_idx, :].astype("float")
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test
        self.train_count = len(training_idx) * (1 - self.validation_split)
        self.val_count = len(training_idx) * self.validation_split
        self.test_count = len(test_idx)

    def get_last_cp_model(self, all_files):
        epoch_re = re.compile(r"weights-improvement-([0-9]+)-")
        max_epoch = 0
        last_icp = ""
        for index, icp_file in enumerate(all_files):
            epoch_flag = epoch_re.search(icp_file)
            epoch = 0  # loss0.4787-valloss0.4075.hdf5a
            if epoch_flag:
                epoch = int(epoch_flag.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                last_icp = icp_file
        return last_icp

    def Train(self):
        None

    def WritePredictionsToFile(self, x_train, y_train, predictions, outfile):
        print("Loading checkpoint file:"+self.icp)
        predictions = np.exp(predictions.reshape((predictions.shape[0],)))
        out_data_hash = {}
        x_train_tr = x_train.transpose()
        for index, hdr in enumerate(self.headers):
            out_data_hash[hdr] = x_train_tr[index].tolist()
        out_data_hash['predicted'] = predictions.tolist()
        if y_train.size != 0:
            y_train = y_train.reshape((y_train.shape[0],))
            error = np.abs(y_train - predictions)
            error_percent = error / y_train
            out_data_hash['original_cost'] = y_train.tolist()
            out_data_hash['error'] = error.tolist()
            out_data_hash['error-percent'] = error_percent.tolist()
            print("Error: "+str(np.mean(error_percent)))
        df = pd.DataFrame(out_data_hash)
        df.to_csv(outfile, index=False, sep=',', encoding='utf-8')
        return None

    # Evalaute model results
    def EvaluateModel(self, all_files, outfile="test-output.csv"):
        None
