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
from sklearn import metrics
from sklearn.linear_model import LassoLars
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os, sys, logging
import time
from baseml import *
import shlex
import pdb
import joblib
import glob

checkpoint_dir = "checkpoints"
class SKlearnRF(BaseMLModel):
    def __init__(self, framework):
        self.framework = framework
        self.config = self.framework.config.GetModel()
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()
        self.step = -1
        self.step_start = framework.args.step_start
        self.step_end = framework.args.step_end
        self.validation_split = float(self.args.validation_split)
        if self.framework.args.validation_split != "":
            self.validation_split = float(self.framework.args.validation_split)
        self.train_test_split = float(self.args.train_test_split)
        if self.framework.args.train_test_split != "":
            self.train_test_split = float(self.framework.args.train_test_split)

    def GetTrainValSplit(self):
        return self.validation_split

    def GetTrainTestSplit(self):
        return self.train_test_split

    def ReadArguments(self):
        arg_string = self.config.ml_arguments
        args = self.parser.parse_args(shlex.split(arg_string))
        return args

    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-rf_random_state", default=0, type=int)
        parser.add_argument("-rf_n_estimators", default=100, type=int)
        parser.add_argument("-rf_crtiterion", default="mse")
        parser.add_argument("-rf_max_depth", default=None, type=int)
        parser.add_argument("-rf_n_jobs", default=None, type=int)
        parser.add_argument("-method", default='RandomForestRegressor')
        parser.add_argument(
            "-real-objective", dest="real_objective", action="store_true"
        )
        parser.add_argument("-alpha", default=0.5, type=float)
        parser.add_argument(
            "-validation-split", dest="validation_split", default="0.20"
        )
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
        exclude_costs,
        parameters_data,
        cost_data,
        samples,
        name="network",
        preload_cost_checkpoints = False
    ):
        BaseMLModel.Initialize(
            self, headers, 
            cost_names,
            valid_costs,
            exclude_costs,
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

        method = RandomForestRegressor
        if self.args.method == 'RandomForestRegressor':
            method = RandomForestRegressor
        elif self.args.method == 'SVR':
            method = SVR 
        elif self.args.method == 'LassoLars':
            method = LassoLars
        for index, cost in enumerate(self.cost_names):
            self.cost_models.append(None)
            if self.IsValidCost(cost):
                self.cost_models[index] = method(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=rf_random_state,
                    n_jobs=rf_n_jobs,
                )
        if preload_cost_checkpoints:
            self.LoadCostCheckPoints()

    # Load the model from the joblib
    def load_model(self, model_name, cost_index):
        model = joblib.load(model_name)
        self.cost_models[cost_index] = model

    def PreLoadData(self):
        BaseMLModel.PreLoadData(self, self.step, self.GetTrainTestSplit(), self.validation_split)
        for index, cost in enumerate(self.cost_names):
            if self.IsValidCost(cost):
                self.PreLoadDataCore(index)

    def PreLoadDataCore(self, cost_index, best=False):
        last_cp = ""
        pattern = "*-cost{}-*weights-improvement*.jbl.lzma".format(cost_index)
        if self.step != self.step_start and not best:
            pattern = "step{}-cost{}-*.jbl.lzma".format(self.prev_step, cost_index)
        all_files = glob.glob(
            os.path.join(checkpoint_dir, pattern)
        )
        last_cp = BaseMLModel.get_last_cp_model(self, all_files)
        if last_cp != "":
            self.load_model(last_cp, cost_index)
            self.disable_icp = True

    # Load checkpoint for all costs
    def LoadCostCheckPoints(self):
        icp_list = self.icp
        re_tag = ".*-cost{}-.*.jbl.lzma"
        for index, cost in enumerate(self.cost_names):
            if not self.IsValidCost(cost):
                continue
            icp = ""
            if index < len(icp_list):
                icp = icp_list[index]
            else:
                files = glob.glob(
                    os.path.join(checkpoint_dir, 
                        re_tag.format(cost_index)))
                icp = BaseMLModel.get_last_cp_model(self, files)
            self.load_model(icp, cost_index)
        
    # Inference on samples, which is type of model specific
    def Inference(self):
        self.cost_models[cost_index].load_weights(self.icp)
        predictions = self.cost_models[cost_index].predict(self.x_train)
        if outfile != None:
            BaseMLModel.WritePredictionsToFile(
                self, self.x_train, self.y_train[cost_index], predictions, outfile
            )
        return predictions.reshape((predictions.shape[0],))

    def TrainCost(self, cost_index=0):
        BaseMLModel.SaveTrainValTestData(self, self.step)
        x_train, y_train, z_train = self.x_train, self.y_train[cost_index].astype(float).reshape((-1)), self.z_train[cost_index].astype(float).reshape((-1))
        x_test, y_test, z_test = self.x_test, self.y_test[cost_index].astype(float).reshape((-1)), self.z_test[cost_index].astype(float).reshape((-1))

        rf_dict = self.rf_dict
        n_estimators = rf_dict["n_estimators"]
        random_state = rf_dict["random_state"]
        crtiterion = rf_dict["crtiterion"]
        max_depth = rf_dict["max_depth"]
        n_jobs = rf_dict["n_jobs"]
        alpha = rf_dict["alpha"]
        rf = self.cost_models[cost_index]

        obj_train = y_train
        if self.args.real_objective:
            obj_train = z_train
        obj_train = obj_train
        start = time.time()
        #pdb.set_trace()
        rf.fit(x_train, obj_train)
        train_count = len(self.x_train)
        test_count = len(self.x_test)
        chk_pnt_name = f"step{self.step}-cost{cost_index}-train{train_count}-val{test_count}-weights-improvement-0.jbl.lzma"
        joblib.dump(rf, os.path.join(checkpoint_dir,chk_pnt_name))
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
            y_train_data = self.compute_error(cost_index, y_train, np.log(np.abs(obj_pred_train)))
            z_train_data = self.compute_error(cost_index, z_train, obj_pred_train)
            if obj_pred_test != None:
                y_test_data = self.compute_error(cost_index, y_test, np.log(np.abs(obj_pred_test)))
                z_test_data = self.compute_error(cost_index, z_test, obj_pred_test)
        else:
            y_train_data = self.compute_error(cost_index, y_train, obj_pred_train)
            if (y_train == z_train).all():
                z_train_data = y_train_data
            else:
                z_train_data = self.compute_error(cost_index, z_train, np.exp(obj_pred_train))
            if obj_pred_test != None:
                y_test_data = self.compute_error(cost_index, y_test, obj_pred_test)
                if (y_train == z_train).all():
                    z_test_data = y_test_datat
                else:
                    z_test_data = self.compute_error(cost_index, z_test, np.exp(obj_pred_test))
        return (
            self.step,
            0,
            y_train_data[0],
            y_test_data[0],
            x_train.shape[0],
            x_test.shape[0],
        )

    def compute_error(self, cost_index, test, pred):
        all_errors = np.abs(test-pred)
        all_errors = np.divide(all_errors, test, out=np.zeros_like(all_errors), where=test!=0)
        mean_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        median_error = np.median(all_errors)
        std_error = np.std(all_errors)
        print(
            "Cost:{} Mean:{} Max:{} Min:{} Median:{} Std:{}".format(
                cost_index, mean_error, max_error, min_error, median_error, std_error
            )
        )
        return [mean_error, max_error, min_error, median_error, std_error]


def GetObject(framework):
    obj = SKlearnRF(framework)
    return obj
