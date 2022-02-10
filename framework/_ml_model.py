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
import os
import numpy as np
import pdb
import argparse
import shlex
from deffe_utils import *

class DeffeMLModel:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetModel()
        self.exclude_costs = self.config.exclude_costs
        self.ml_model_script = LoadModule(self.framework, self.config.ml_model_script)
        self.accuracy = (0.0, 0.0)

    def IsValidCost(self, cost):
        if len(self.valid_costs) > 0:
            if cost not in self.valid_costs:
                return False
        return True

    # Initialize the members
    def Initialize(self, cost_names, valid_costs):
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()
        self.parameters = np.array([])
        self.cost_output = np.array([])
        self.samples = np.array([])
        self.cost_names = cost_names
        self.valid_costs = valid_costs
            

    # Read arguments provided in JSON configuration file
    def ReadArguments(self):
        arg_string = self.config.arguments
        args = self.parser.parse_args(shlex.split(arg_string))
        return args

    # Add command line arguments to parser
    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        return parser

    # Check if model is ready
    def IsModelReady(self):
        # TODO: This method is yet to be implemented
        return False

    # Get Train-Validation split
    def GetTrainValSplit(self):
        return self.ml_model_script.GetTrainValSplit()

    # Get Train-Validation split
    def GetTrainTestSplit(self):
        return self.ml_model_script.GetTrainTestSplit()

    # Initialize model parameters and costs
    def InitializeSamples(self, samples, headers, cost_names, params, cost_data=None, step=0, expand_list=True, preload_cost_checkpoints=False):
        params_valid_indexes = []
        cost_metrics = []
        indexes = samples
        if type(params) == list:
            params = np.array(params)
        if cost_data != None:
            cost_metrics = []
            for index, (flag, eval_type, actual_cost, run_dir) in enumerate(cost_data):
                if flag == self.framework.valid_flag:
                    params_valid_indexes.append(index)
                    cost_metrics.append(actual_cost)
            if len(params_valid_indexes) == 0:
                print("[Warning] no samples to train in this step !")
        else:
            params_valid_indexes = range(len(indexes))
        cost_metrics = np.array(cost_metrics)
        valid_indexes = np.array(samples)[params_valid_indexes,]
        if len(self.parameters) == 0 or not expand_list:
            self.parameters = params[params_valid_indexes,]
            self.cost_output = cost_metrics
            self.samples = valid_indexes
        else:
            self.parameters = np.append(self.parameters, 
                    params[params_valid_indexes,], axis=0)
            self.cost_output = np.append(self.cost_output, 
                    cost_metrics, axis=0)
            self.samples = np.append(self.samples, 
                    valid_indexes, axis=0)
        self.params_valid_indexes=params_valid_indexes
        #print("Headers:{}".format(headers))
        self.ml_model_script.Initialize(
            step,
            headers,
            cost_names,
            self.valid_costs,
            self.exclude_costs,
            self.parameters,
            self.cost_output,
            self.samples,
            preload_cost_checkpoints = preload_cost_checkpoints 
        )
        self.ml_model_script.PreLoadData()

    # Run the prediction/inference
    def Inference(self):
        all_output = self.ml_model_script.Inference()
        return all_output.tolist()

    # Train the model
    def Train(self, threading_model=False):
        self.accuracy = self.ml_model_script.Train(True)
        return self.accuracy

    # Evaluate model results
    def EvaluateModel(self, all_files, outfile="evaluate-model-output.csv"):
        self.ml_model_script.EvaluateModel(all_files, outfile)

    # Preload the pretrained model
    def PreLoadData(self):
        self.ml_model_script.PreLoadData()


def GetObject(*args):
    obj = DeffeMLModel(*args)
    return obj
