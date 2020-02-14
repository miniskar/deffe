import os

class DeffeFullExploration:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetModel()

    def IsCompleted(self):
        return self.framework.sampling.IsCompleted()

    def IsModelReady(self):
        if self.framework.model.IsModelReady():
            return True
        return False

    def Explore(self):
        None

def GetObject(framework):
    obj = DeffeFullExploration(framework)
    return obj
