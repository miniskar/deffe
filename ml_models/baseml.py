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
import pdb
import re
import os
import pandas as pd
from deffe_thread import *
from deffe_utils import Log, ReshapeCosts, IsNumber, IsStringNumber

checkpoint_dir = "checkpoints"
class BaseMLModel:
    def __init__(self):
        None

    def Initialize(self, headers, config_cost_names, cost_names, 
            valid_costs, exclude_costs,
            parameters, cost_data, samples, cost_scaling_factor):
        Log("Headers: " + str(headers))
        orig_cost_data = cost_data
        self.headers = headers
        self.config_cost_names = config_cost_names
        self.cost_names = cost_names
        self.valid_costs = valid_costs
        self.exclude_costs = exclude_costs.copy()
        #pdb.set_trace()
        self.parameters_data = parameters
        self.cost_data = cost_data
        self.orig_cost_data = orig_cost_data
        self.cost_name_index = {}
        for index, cost in enumerate(self.config_cost_names):
            self.cost_name_index[cost] = index
            if cost not in self.cost_names:
                self.exclude_costs.append(cost)
        if cost_data.size > 0:
            for index, cost in enumerate(self.config_cost_names):
                #print(f"Index:{index} Cost:{cost} to be excluded from ML")
                if self.IsValidCost(cost):
                    #print(f"Valid Index:{index} Cost:{cost} to be excluded from ML")
                    #pdb.set_trace()
                    cost_data_col = cost_data[:,index]
                    if cost_data_col.size > 0 and not (IsNumber(cost_data_col[0]) or IsStringNumber(cost_data_col[0])):
                        self.exclude_costs.append(cost)
                    else:
                        self.cost_data[:,index] = self.cost_data[:,index].astype('float') * cost_scaling_factor
                        self.orig_cost_data[:,index] = orig_cost_data[:,index].astype('float') * cost_scaling_factor 
                    #print(f"Invalid cost:{cost} to be excluded from ML values:{cost_data[:,index]}")
        self.sample_actual_indexes = samples
        self.cost_models = []

    def IsValidCost(self, cost):
        if len(self.valid_costs) > 0:
            if cost not in self.valid_costs:
                return False
        if cost in self.exclude_costs:
            return False
        return True

    def GetModel(self, index):
        if index < len(self.cost_models):
            return self.cost_models[index]
        return None

    def Train(self, threading_model=False):
        output = []
        valid_trains = []
        valid_costs = []
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #pdb.set_trace()
        #self.SaveTrainValTestData(self.step)
        for cost in self.config_cost_names:
            index = self.cost_name_index[cost]
            output.append(None)
            if self.IsValidCost(cost):
                valid_trains.append((index, cost))
                valid_costs.append(cost)
        train_threads_list = []
        #pdb.set_trace()
        print(f"Valid cost metrics for training: {valid_costs}")
        for (index, cost) in valid_trains:
            if threading_model:
                def TrainThread(self, index, cost):
                    out = self.TrainCost(index)
                    output[index] = out
                train_th_obj = DeffeThread(TrainThread, 
                        (self, index, cost), True)
                train_th_obj.StartThread()
                train_threads_list.append(train_th_obj)
            else:
                output[index] = self.TrainCost(index)
        for th in train_threads_list:
            th.JoinThread()
        return output

    def PreLoadData(self, step, train_test_split, validation_split, shuffle=False):
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        train_count = int(parameters.shape[0] * train_test_split)
        test_count = parameters.shape[0] - train_count
        #print("Init Total count:" + str(parameters.shape[0]))
        #print("Init Train count:" + str(train_count))
        #print("Init Test count:" + str(test_count))
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
        x_train = parameters[training_idx, :].astype("float")
        x_test = parameters[test_idx, :].astype("float")
        y_train = np.array([])
        z_train = np.array([])
        y_test = np.array([])
        z_test = np.array([])
        if cost_data.size != 0:
            y_train = cost_data[training_idx, :]
            z_train = orig_cost_data[training_idx, :]
            y_test = cost_data[test_idx, :]
            z_test = orig_cost_data[test_idx, :]
            y_train = ReshapeCosts(y_train)
            z_train = ReshapeCosts(z_train)
            y_test = ReshapeCosts(y_test)
            z_test = ReshapeCosts(z_test)
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def SaveTrainValTestData(self, cost_index=0, step=-1):
        if step == -1:
            print("Saving ML model indices: ml-indices.npy")
            np.save(f"ml-cost{cost_index}-indices.npy", 
                    self.sample_actual_indexes)
        else:
            print("Saving ML indices: "
                    "step{}-cost{}-ml-indices.npy".format(step, cost_index))
            np.save("step{}-cost{}-ml-indices.npy".format(step, cost_index), 
                    self.sample_actual_indexes)

    def LoadTrainValTestData(self, step=-1):
        if step == -1:
            print("Loading ML indices: ml-indices.npy")
            sample_load_indexes = np.load(
                    "ml-indices.npy")
        else:
            print("Loading ML indices: "
                    "step{}-ml-indices.npy".format(step))
            sample_load_indexes = np.load(
                    "step{}-ml-indices.npy".format(step))
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        train_val_indexes = self.sample_actual_indexes
        index_hash = {
            target_index: index for index, target_index in enumerate(train_val_indexes)
        }
        train_val_indexes = sample_load_indexes
        training_idx = [index_hash[index] for index in train_val_indexes]
        self.x_all = parameters.astype("float")
        self.y_all = cost_data.astype("float")
        x_train = parameters[training_idx, :].astype("float")
        y_train = cost_data[training_idx, :].astype("float")
        z_train = orig_cost_data[training_idx, :].astype("float")
        # Get all remaining data other than traininga
        all_indexes = range(parameters.shape[0])
        test_idx = np.delete(all_indexes, training_idx)
        x_test = parameters[test_idx, :].astype("float")
        y_test = cost_data[test_idx, :].astype("float")
        z_test = orig_cost_data[test_idx, :].astype("float")
        y_train = ReshapeCosts(y_train)
        z_train = ReshapeCosts(z_train)
        y_test = ReshapeCosts(y_test)
        z_test = ReshapeCosts(z_test)
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test
        self.train_count = len(training_idx) * (1 - self.validation_split)
        self.val_count = len(training_idx) * self.validation_split
        self.test_count = len(test_idx)

    def get_last_cp_model(self, all_files):
        epoch_re = re.compile(r"step([0-9]+).*weights-improvement-([0-9]+)-")
        max_epoch = -1
        last_icp = ""
        for index, icp_file in enumerate(all_files):
            epoch_flag = epoch_re.search(icp_file)
            epoch = 0  # loss0.4787-valloss0.4075.hdf5a
            step = 0  # loss0.4787-valloss0.4075.hdf5a
            if epoch_flag:
                step = int(epoch_flag.group(1))
                epoch = int(epoch_flag.group(2))
            step_epoch = step * 10000000 + epoch
            if step_epoch > max_epoch:
                max_epoch = step_epoch
                last_icp = icp_file
        return last_icp

    # Evalaute model results
    def EvaluateModel(self, all_files, outfile="test-output.csv"):
        None
