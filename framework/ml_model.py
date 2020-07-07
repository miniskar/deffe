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
        self.ml_model_script = LoadModule(self.framework, self.config.ml_model_script)
        self.accuracy = (0.0, 0.0)

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
    def InitializeSamples(self, samples, headers, params, cost=None, step=0):
        params_valid_indexes = []
        cost_metrics = []
        indexes = samples
        if cost != None:
            cost_metrics = []
            for index, (flag, eval_type, actual_cost) in enumerate(cost):
                if flag == self.framework.valid_flag:
                    params_valid_indexes.append(index)
                    cost_metrics.append(actual_cost)
            if len(params_valid_indexes) == 0:
                print("[Warning] no samples to train in this step !")
        else:
            params_valid_indexes = range(len(indexes))
        cost_metrics = np.array(cost_metrics)
        valid_indexes = np.array(samples)[params_valid_indexes,]
        if len(self.parameters) == 0:
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
        self.ml_model_script.Initialize(
            step,
            headers,
            self.cost_names,
            self.valid_costs,
            self.parameters,
            self.cost_output,
            self.samples
        )
        self.ml_model_script.PreLoadData()

    # Run the prediction/inference
    def Inference(self, output_file=""):
        cost_index = 0
        all_output = self.ml_model_script.Inference(cost_index, output_file)
        cost = []
        for output in all_output:
            cost.append(
                (self.framework.valid_flag, self.framework.predicted_flag, output)
            )
        return cost

    # Train the model
    def Train(self, threading_model=False):
        self.accuracy = self.ml_model_script.Train(True)
        return self.accuracy

    # Evaluate model results
    def EvaluateModel(self, all_files, outfile="test-output.csv"):
        self.ml_model_script.EvaluateModel(all_files, outfile)

    # Preload the pretrained model
    def PreLoadData(self):
        self.ml_model_script.PreLoadData()


def GetObject(framework):
    obj = DeffeMLModel(framework)
    return obj
