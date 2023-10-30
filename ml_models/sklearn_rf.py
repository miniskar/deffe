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
from deffe_utils import ReplaceInfiniteWithMax
import os, sys, logging
import time
from baseml import *
import shlex
import pdb
import joblib
import glob
import warnings 
# suppress warnings 
warnings.filterwarnings('ignore') 

checkpoint_dir = "checkpoints"
class SKlearnRF(BaseMLModel):
    def __init__(self, framework):
        self.framework = framework
        self.config = self.framework.config.GetModel()
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()
        self.prev_step = -1
        self.step = -1
        self.step_start = framework.args.step_start
        self.step_end = framework.args.step_end
        self.icp = self.args.icp
        if len(self.framework.args.icp) > 0:
            self.icp = self.framework.args.icp
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
        parser.add_argument("-icp", dest="icp", nargs='*', default=[])
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
        config_cost_names,
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
            config_cost_names,
            cost_names,
            valid_costs,
            exclude_costs,
            parameters_data, cost_data, samples, 1.0
        )
        args = self.args
        self.prev_step = self.step
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
        for index, cost in enumerate(self.config_cost_names):
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
        #pdb.set_trace()
        #Log("Trying to Load checkpoint file: " + model_name)
        if os.path.exists(model_name):
            Log("Loading checkpoint file: " + model_name)
            model = joblib.load(model_name)
            self.cost_models[cost_index] = model
            return True
        return False

    def PreLoadData(self):
        BaseMLModel.PreLoadData(self, self.step, self.GetTrainTestSplit(), self.validation_split)
        for index, cost in enumerate(self.config_cost_names):
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
        #pdb.set_trace()
        last_cp = BaseMLModel.get_last_cp_model(self, all_files)
        #print(f"Last best cp for cost:{cost_index} cp:{last_cp}")
        if last_cp != "":
            #pdb.set_trace()
            return self.load_model(last_cp, cost_index)
        return False

    # Load checkpoint for all costs
    def LoadCostCheckPoints(self):
        icp_list = self.icp
        re_tag = ".*-cost{}-.*.jbl.lzma"
        #pdb.set_trace()
        for index, cost in enumerate(self.config_cost_names):
            if not self.IsValidCost(cost):
                continue
            icp = ""
            if index < len(icp_list) and icp_list[index] != '' and os.path.exists(icp_list[index]):
                icp = icp_list[index]
            else:
                files = glob.glob(
                    os.path.join(checkpoint_dir, 
                        re_tag.format(index)))
                icp = BaseMLModel.get_last_cp_model(self, files)
            self.load_model(icp, index)
        
    # Inference on samples, which is type of model specific
    def Inference(self):
        icp_list = self.icp
        all_predictions = []
        for index, cost in enumerate(self.config_cost_names):
            if not self.IsValidCost(cost):
                all_predictions.append(None)
                continue
            if self.PreLoadDataCore(index, best=True):
                predictions = self.cost_models[index].predict(self.x_train)
                all_predictions.append(predictions.reshape((predictions.shape[0],)).tolist())
            else:
                all_predictions.append(None)
        return all_predictions

    def TrainCost(self, cost_index=0):
        BaseMLModel.SaveTrainValTestData(self, cost_index, self.step)
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
    
        obj_train = ReplaceInfiniteWithMax(obj_train) 
        rf.fit(x_train, obj_train)
        train_count = len(self.x_train)
        test_count = len(self.x_test)
        chk_pnt_name = f"step{self.step}-cost{cost_index}-train{train_count}-val{test_count}-weights-improvement-0-sk.jbl.lzma"
        model_file = os.path.join(checkpoint_dir,chk_pnt_name)
        joblib.dump(rf, model_file)
        print(f"Saving the model to file:{model_file}")
        #pdb.set_trace()
        if len(self.icp) <= cost_index:
            for i in range(len(self.icp),cost_index+1, 1):
                self.icp.append('')
        self.icp[cost_index] = chk_pnt_name
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

    def compute_error(self, cost_index, test, pred, is_exp=False, is_log=False):
        error_diff = ReplaceInfiniteWithMax(np.abs(test-pred))
        all_errors = np.divide(error_diff, test, out=np.zeros_like(error_diff), where=test!=0)
        all_errors = all_errors[~np.isnan(all_errors)]
        mean_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        median_error = np.median(all_errors)
        std_error = np.std(all_errors)
        print(
            "Cost:{} Mean:{} Max:{} Min:{} Median:{} Std:{} Size:{}".format(
                cost_index, mean_error, max_error, 
                min_error, median_error, std_error, len(test)
            )
        )
        return [mean_error, max_error, min_error, median_error, std_error]


def GetObject(framework):
    obj = SKlearnRF(framework)
    return obj
