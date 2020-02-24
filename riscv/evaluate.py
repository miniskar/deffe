import os
import numpy as np
import pdb

class DeffeRISCVEvaluate:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetEvaluate()

    def Run(self, parameters):
        eval_output = np.zeros(shape=(parameters.shape[0], 1))
        return eval_output

def GetObject(framework):
    obj = DeffeRISCVEvaluate(framework)
    return obj
