import os

class DeffeRISCVExtract:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetExtract()

    def Run(self, eval_output):
        return eval_output

def GetObject(framework):
    obj = DeffeRISCVExtract(framework)
    return obj


