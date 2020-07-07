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
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLars
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os, sys, logging
import time
from baseml import *
import shlex
import pdb


class SKlearnRF(BaseMLModel):
    def __init__(self, framework):
        self.framework = framework
        self.config = self.framework.config.GetModel()
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()
        self.step = -1
        self.step_start = framework.args.step_start
        self.step_end = framework.args.step_end

    def GetTrainValSplit(self):
        return float(self.args.train_test_split)

    def GetTrainTestSplit(self):
        return float(self.args.train_test_split)

    def ReadArguments(self):
        arg_string = self.config.ml_arguments
        args = self.parser.parse_args(shlex.split(arg_string))
        return args

    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-rf_random_state", default=0, type=int)
        parser.add_argument("-rf_n_estimators", default=10, type=int)
        parser.add_argument("-rf_crtiterion", default="mse")
        parser.add_argument("-rf_max_depth", default=None, type=int)
        parser.add_argument("-rf_n_jobs", default=None, type=int)
        parser.add_argument(
            "-real-objective", dest="real_objective", action="store_true"
        )
        parser.add_argument("-alpha", default=0.5, type=float)
        parser.add_argument(
            "-train-test-split", dest="train_test_split", default="1.00"
        )
        return parser

    def Initialize(
        self,
        step,
        headers,
        cost_names,
        valid_costs,
        parameters_data,
        cost_data,
        samples,
        name="network",
    ):
        BaseMLModel.Initialize(
            self, headers, 
            cost_names,
            valid_costs,
            parameters_data, cost_data, samples, 1.0
        )
        args = self.args
        self.step = step
        self.headers = headers
        print("Headers: " + str(headers))
        self.parameters_data = parameters_data
        self.cost_data = cost_data
        self.orig_cost_data = cost_data
        rf_random_state = args.rf_random_state
        rf_n_estimators = args.rf_n_estimators
        rf_crtiterion = args.rf_crtiterion
        rf_max_depth = args.rf_max_depth
        rf_n_jobs = args.rf_n_jobs
        alpha = args.alpha

        rf_dict = {}
        rf_dict["random_state"] = rf_random_state
        rf_dict["n_estimators"] = rf_n_estimators
        rf_dict["crtiterion"] = rf_crtiterion
        rf_dict["max_depth"] = rf_max_depth
        rf_dict["n_jobs"] = rf_n_jobs
        rf_dict["alpha"] = alpha
        self.rf_dict = rf_dict
        self.model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=rf_random_state,
            n_jobs=rf_n_jobs,
        )

    def PreLoadData(self):
        BaseMLModel.PreLoadData(self, self.step, self.GetTrainTestSplit(), 0.20)
        x_train, y_train, z_train = self.x_train, self.y_train, self.z_train
        x_test, y_test, z_test = self.x_test, self.y_test, self.z_test
        y_train = np.log(y_train.reshape((y_train.shape[0],)))
        z_train = z_train.reshape((z_train.shape[0],))
        y_test = np.log(y_test.reshape((y_test.shape[0],)))
        z_test = z_test.reshape((z_test.shape[0],))
        self.y_train = y_train
        self.z_train = z_train
        self.y_test = y_test
        self.z_test = z_test

    # Inference on samples, which is type of model specific
    def Inference(self, cost_index, outfile=""):
        self.model.load_weights(self.icp)
        predictions = self.model.predict(self.x_train)
        if outfile != None:
            BaseMLModel.WritePredictionsToFile(
                self, self.x_train, self.y_train, predictions, outfile
            )
        return predictions.reshape((predictions.shape[0],))

    def TrainCost(self, cost_index=0):
        BaseMLModel.SaveTrainValTestData(self, self.step)
        x_train, y_train, z_train = self.x_train, self.y_train, self.z_train
        x_test, y_test, z_test = self.x_test, self.y_test, self.z_test

        rf_dict = self.rf_dict
        n_estimators = rf_dict["n_estimators"]
        random_state = rf_dict["random_state"]
        crtiterion = rf_dict["crtiterion"]
        max_depth = rf_dict["max_depth"]
        n_jobs = rf_dict["n_jobs"]
        alpha = rf_dict["alpha"]
        rf = self.model

        obj_train = y_train
        if self.args.real_objective:
            obj_train = z_train
        start = time.time()
        rf.fit(x_train, obj_train)
        lapsed_time = "{:.3f} seconds".format(time.time() - start)
        print("Total runtime of script: " + lapsed_time)

        print("Regression training complete...")

        start = time.time()
        obj_pred_train = rf.predict(x_train)
        obj_pred_test = None
        if len(x_test) > 0:
            obj_pred_test = rf.predict(x_test)
        inference_time = "{:.3f} seconds".format(time.time() - start)
        print("Total runtime of script: " + inference_time)

        print("Regression prediction complete...")

        y_train_data = z_train_data = y_test_data = z_test_data = [0.0]
        if self.args.real_objective:
            y_train_data = self.compute_error(y_train, np.log(np.abs(obj_pred_train)))
            z_train_data = self.compute_error(z_train, obj_pred_train)
            if obj_pred_test != None:
                y_test_data = self.compute_error(y_test, np.log(np.abs(obj_pred_test)))
                z_test_data = self.compute_error(z_test, obj_pred_test)
        else:
            y_train_data = self.compute_error(y_train, obj_pred_train)
            z_train_data = self.compute_error(z_train, np.exp(obj_pred_train))
            if obj_pred_test != None:
                y_test_data = self.compute_error(y_test, obj_pred_test)
                z_test_data = self.compute_error(z_test, np.exp(obj_pred_test))
        return (
            self.step,
            0,
            y_train_data[0],
            y_test_data[0],
            x_train.shape[0],
            x_test.shape[0],
        )

    def compute_error(self, test, pred):
        all_errors = np.zeros(pred.shape)
        for i in range(len(test)):
            out_val = test[i]
            pred_val = pred[i]
            error_percentage = np.abs(out_val - pred_val) / out_val
            all_errors[i] = error_percentage
        mean_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        median_error = np.median(all_errors)
        std_error = np.std(all_errors)
        print(
            "Mean:{} Max:{} Min:{} Median:{} Std:{}".format(
                mean_error, max_error, min_error, median_error, std_error
            )
        )
        return [mean_error, max_error, min_error, median_error, std_error]


def GetObject(framework):
    obj = SKlearnRF(framework)
    return obj
