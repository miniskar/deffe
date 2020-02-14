import os

class DeffeMLModel:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetModel()

    def IsModelReady(self):
        return False

    def Inference(self, samples):
        return self.batch_output

    def Train(self, samples, eval_output):
        None

def GetObject(framework):
    obj = DeffeMLModel(framework)
    return obj
