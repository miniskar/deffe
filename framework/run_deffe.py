import glob, importlib, os, pathlib, sys
import socket

framework_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, framework_path)
sys.path.insert(0, os.path.join(framework_path, "utils"))
from read_config import *
from workload_excel import *
from parameters import *
import numpy as np

# Requirements
# * Scenario based evaluation and ml model creation
# * Support multi-scenario exploration
# * Multi-cost exploration support
# * Support of partial set of parameters in the exploration
#   ** Example: Some kernels are limited to some cores
# * Mapping of knobs to common parameters in exploration
#   ** Example: Mapping of core0_l1d_size and core1_l1d_size to l1d_size parameter in the evaluation and ml-model 
class DeffeFramework:
    def __init__(self, args):
        config = DeffeConfig(args.config)
        self.config = config
        self.args = args
        self.init_n_train = 100
        self.init_n_val = 2 * self.init_n_train
        self.Initialize()
        self.parameters = Parameters(config)
        self.model = self.LoadModule(config.GetModel().script)
        self.sampling = self.LoadModule(config.GetSampling().script)
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
        for explore_groups in self.config.GetExploration().exploration_list:
            (param_list, pruned_param_list, n_samples)  = self.parameters.Initialize(explore_groups)
            headers = self.parameters.GetHeaders(param_list)
            pruned_headers = self.parameters.GetHeaders(pruned_param_list)
            self.sampling.Initialize(n_samples, self.init_n_train, self.init_n_val)
            step = 0
            while(not self.exploration.IsCompleted()):
                print("***** Step {} *****".format(step))
                samples = self.sampling.GetBatch()
                parameters = self.parameters.GetParameters(samples, param_list)
                parameters_normalize = self.parameters.GetParameters(samples, pruned_param_list, with_indexing=True, with_normalize=False)
                if self.exploration.IsModelReady():
                    batch_output = self.model.Inference(parameters_normalize)
                else:
                    eval_output = self.evaluate.Run(parameters)
                    batch_output = self.extract.Run(eval_output)
                    (train_acc, val_acc) = self.model.Train(pruned_headers, parameters_normalize, batch_output)
                    print("Train accuracy: "+str(train_acc)+" Val accuracy: "+str(val_acc))
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
