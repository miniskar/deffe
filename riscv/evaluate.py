import os

class DeffeRISCVEvaluate:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetEvaluate()

    def Run(self, samples):
        eval_output = None
        return eval_output

def GetObject(framework):
    obj = DeffeRISCVEvaluate(framework)
    return obj
