import os
import numpy as np

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

    def Train(self, headers, params, cost):
        self.ml_model_script.Initialize(headers, params, cost)
        self.ml_model_script.preprocess_data()
        return self.ml_model_script.Train()

def GetObject(framework):
    obj = DeffeMLModel(framework)
    return obj
