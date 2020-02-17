import os

class DeffeMLModel:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetModel()
        self.ml_model_script = framework.LoadModule(self.config.ml_model_script)
        parameters = []
        cost_output = []
        self.ml_model_script.Initialize(parameters, cost_output, cost_output)

    def IsModelReady(self):
        return False

    def Inference(self, samples):
        return self.batch_output

    def GetParametersCost(self, samples, eval_output):
        return (params, cost)

    def Train(self, samples, eval_output):
        (params, cost) = self.GetParametersCost(samples, eval_output)
        self.ml_model_script.Initialize(params, cost, cost)
        self.ml_model_script.
        self.ml_model_script.Train()

def GetObject(framework):
    obj = DeffeMLModel(framework)
    return obj
