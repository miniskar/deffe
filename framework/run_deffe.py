import glob, importlib, os, pathlib, sys
import socket

framework_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, framework_path)
sys.path.insert(0, os.path.join(framework_path, "utils"))
from read_config import *
from workload_excel import *
from params_cost import *
import numpy as np

class DeffeFramework:
    def __init__(self, args):
        config = DeffeConfig(args.config)
        self.config = config
        self.args = args
        self.init_n_train = 100
        self.init_n_val = 2 * self.init_n_train
        self.Initialize()
        self.params_cost = ParamsCost(config)
        self.model = self.LoadModule(config.GetModel().script)
        self.sampling = self.LoadModule(config.GetSampling().script)
        self.sampling.Initialize(np.arange(len(self.params_cost.all_output)), self.init_n_train, self.init_n_val)
        self.exploration = self.LoadModule(config.GetExploration().script)
        self.evaluate = self.LoadModule(config.GetEvaluate().script)
        self.extract = self.LoadModule(config.GetExtract().script)

    def Configure(self):
        fr_config = self.config.GetFramework()
        #TODO: set parameters

    def Initialize(self):
        python_paths = self.config.GetPythonPaths()
        for path in python_paths:
            if path[0] != '/':
                sys.path.insert(0, os.path.join(os.getcwd(), path))
            else:
                sys.path.insert(0, path)
        print(sys.path)
        self.Configure()

    def LoadModule(self, py_file):
        py_mod_name = pathlib.Path(py_file).stem
        py_mod = importlib.import_module(py_mod_name)
        return py_mod.GetObject(self)

    def Run(self):
        step = 0
        while(not self.exploration.IsCompleted()):
            print("***** Step {} *****".format(step))
            samples = self.sampling.GetBatch()
            if self.exploration.IsModelReady():
                batch_output = self.model.Inference(samples)
            else:
                eval_output = self.evaluate.Run(samples)
                batch_output = self.extract.Run(eval_output)
                self.model.Train(samples, batch_output)
            self.sampling.Step()
            step = step + 1

def InitParser(parser):
    parser.add_argument('-config', dest='config', default="config.json")
    
def main(args):
    framework = DeffeFramework(args)
    framework.Run()

if __name__ == "__main__":
    print("Current directory: "+os.getcwd())
    print("Machine: "+socket.gethostname())
    parser = argparse.ArgumentParser()
    InitParser(parser)
    args = parser.parse_args()
    start = time.time()
    main(args)
    lapsed_time = "{:.3f} seconds".format(time.time() - start)
    print("Total runtime of script: "+lapsed_time)
