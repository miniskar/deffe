import os
import numpy as np
import pdb

class DeffeMLModel:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetModel()
        self.ml_model_script = framework.LoadModule(self.config.ml_model_script)
        parameters = np.array([])
        cost_output = np.array([])

    def IsModelReady(self):
        return False

    def Inference(self, samples):
        return self.batch_output

    def GetParametersCost(self, parameters, eval_output):
        (train_idx, val_idx) = samples
        params = train + val
        return (params, cost)

    def Train(self, step, headers, params, cost):
        params_valid_indexes = []
        cost_metrics = []
        for index, (flag, actual_cost) in enumerate(cost):
            if flag == self.framework.valid_flag:
                params_valid_indexes.append(index)
                cost_metrics.append(actual_cost)
        self.ml_model_script.Initialize(step, headers, params[params_valid_indexes,], np.array(cost_metrics))
        self.ml_model_script.preprocess_data()
        return self.ml_model_script.Train()

def GetObject(framework):
    obj = DeffeMLModel(framework)
    return obj
