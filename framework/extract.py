import os

class DeffeExtract:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetExtract()

    def InitializeParser(parser):
        None

    def SetArgs(args):
        self.args = args

    def Run(self, param_list, eval_output):
        return eval_output

def GetObject(framework):
    obj = DeffeExtract(framework)
    return obj


