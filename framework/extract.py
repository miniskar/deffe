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
        batch_output = []
        for (flag, output) in eval_output:
            if flag == self.framework.predicted_flag:
                batch_output.append((self.framework.valid_flag, output))
            else:
                batch_output.append((self.framework.not_valid_flag, [0,]))
        return batch_output

def GetObject(framework):
    obj = DeffeExtract(framework)
    return obj


