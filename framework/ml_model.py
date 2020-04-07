## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
 # @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
 #         miniskarnr@ornl.gov
 # 
 # Modification:
 #              Baseline code
 # Date:        Apr, 2020
 #**************************************************************************
###
import os
import numpy as np
import pdb
import argparse
import shlex

class DeffeMLModel:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetModel()
        self.ml_model_script = framework.LoadModule(self.config.ml_model_script)
        self.accuracy = (0.0, 0.0)
        parameters = np.array([])
        cost_output = np.array([])

    # Initialize the members
    def Initialize(self):
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()

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

    # Run the prediction/inference
    def Inference(self, samples):
        # TODO: Inference is yet to be implemented
        return self.batch_output

    # Train the model
    def Train(self, step, headers, params, cost):
        params_valid_indexes = []
        cost_metrics = []
        for index, (flag, eval_type, actual_cost) in enumerate(cost):
            if flag == self.framework.valid_flag:
                params_valid_indexes.append(index)
                cost_metrics.append(actual_cost)
        if len(params_valid_indexes) == 0:
            print("[Warning] no samples to train in this step !")
            return self.accuracy
        self.ml_model_script.Initialize(step, headers, params[params_valid_indexes,], np.array(cost_metrics))
        self.ml_model_script.preprocess_data()
        self.accuracy = self.ml_model_script.Train()
        return self.accuracy

def GetObject(framework):
    obj = DeffeMLModel(framework)
    return obj
