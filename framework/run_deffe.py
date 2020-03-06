import glob, importlib, os, pathlib, sys
import socket
import pdb
import signal

framework_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#sys.path.insert(0, framework_path)
framework_env = os.getenv('DEFFE_DIR')
if framework_env  == None: 
    os.environ['DEFFE_DIR'] = framework_path
sys.path.insert(0, os.getenv('DEFFE_DIR'))
sys.path.insert(0, os.path.join(framework_path, "utils"))
sys.path.insert(0, os.path.join(framework_path, "ml_models"))
sys.path.insert(0, os.path.join(framework_path, "framework"))
from read_config import *
from parameters import *
import numpy as np
from workload_excel import *

# Requirements
# * Scenario based evaluation and ml model creation
# * Support multi-scenario exploration
# * Multi-cost exploration support
# * Support of partial set of parameters in the exploration
#   ** Example: Some kernels are limited to some cores
# * Mapping of knobs to common parameters in exploration
#   ** Example: Mapping of core0_l1d_size and core1_l1d_size to l1d_size parameter in the evaluation and ml-model 
# * Should support prediction and exploration on pre-evaluated data
class DeffeFramework:
    def __init__(self, args):
        self.predicted_flag = 0
        self.evaluate_flag = 1
        self.valid_flag = 1
        self.not_valid_flag = 0
        config = DeffeConfig(args.config)
        self.config = config
        self.fr_config = self.config.GetFramework()
        self.args = args
        self.init_n_train = 100
        self.init_n_val = 2 * self.init_n_train
        self.InitializePythonPaths()
        self.parameters = Parameters(config, self)
        self.model = self.LoadModule(config.GetModel().pyscript)
        self.sampling = self.LoadModule(config.GetSampling().pyscript)
        self.exploration = self.LoadModule(config.GetExploration().pyscript)
        self.evaluate = self.LoadModule(config.GetEvaluate().pyscript)
        self.extract = self.LoadModule(config.GetExtract().pyscript)
        self.slurm = self.LoadModule(config.GetSlurm().pyscript)
        self.exploration_table = Workload()
        self.evaluation_table = Workload()
        self.ml_predict_table = Workload()
        self.evaluation_predict_table = Workload()
 
    # Initialize the python paths        
    def InitializePythonPaths(self):
        python_paths = self.config.GetPythonPaths()
        for path in python_paths:
            if path[0] != '/':
                sys.path.insert(0, os.path.join(os.getcwd(), path))
            else:
                sys.path.insert(0, path)
        #print(sys.path)

    def LoadModule(self, py_file):
        py_mod_name = pathlib.Path(py_file).stem
        py_mod = importlib.import_module(py_mod_name)
        return py_mod.GetObject(self)

    def WriteExplorationOutput(self, parameter_values, batch_output):
        for index, (valid_flag, eval_type, cost_metrics) in enumerate(batch_output):
            param_val = parameter_values[index].tolist()
            cost_metrics = cost_metrics.tolist()
            if eval_type == self.evaluate_flag:
                self.evaluation_table.WriteDataInCSV(param_val + cost_metrics)
                self.evaluation_predict_table.WriteDataInCSV(param_val + cost_metrics)
            if eval_type == self.predicted_flag:
                self.ml_predict_table.WriteDataInCSV(param_val + cost_metrics)
                self.evaluation_predict_table.WriteDataInCSV(param_val + cost_metrics)

    def Run(self):
        if not os.path.exists(self.fr_config.run_directory):
            os.makedirs(self.fr_config.run_directory)
        for explore_groups in self.config.GetExploration().exploration_list:
            (param_list, pruned_param_list, n_samples)  = self.parameters.Initialize(explore_groups.groups)
            headers = self.parameters.GetHeaders(param_list)
            pruned_headers = self.parameters.GetHeaders(pruned_param_list)
            self.evaluate.Initialize(param_list, pruned_param_list, self.config.GetCosts(), explore_groups.pre_evaluated_data)
            self.extract.Initialize(param_list, self.config.GetCosts())
            self.config.WriteFile(explore_groups.name+"-minmax.json", self.parameters.GetMinMaxToJSonData())
            self.evaluation_table.WriteHeaderInCSV(explore_groups.evaluation_table, self.evaluate.param_hdrs+self.config.GetCosts())
            self.ml_predict_table.WriteHeaderInCSV(explore_groups.ml_predict_table, self.evaluate.param_hdrs+self.config.GetCosts())
            self.evaluation_predict_table.WriteHeaderInCSV(explore_groups.evaluation_predict_table, self.evaluate.param_hdrs+self.config.GetCosts())
            if self.args.only_preloaded_data_exploration:
                n_samples = len(self.evaluate.param_data_hash) 
            self.sampling.Initialize(n_samples, self.init_n_train, self.init_n_val)
            step = 0
            inc = int(self.args.step_inc)
            while(not self.exploration.IsCompleted()):
                if step != 0:
                    self.sampling.StepWithInc(inc)
                if self.args.step_start != '':
                    if step < int(self.args.step_start):
                        step = step + 1
                        continue
                if self.args.step_end != '':
                    if step >= int(self.args.step_end):
                        step = step + 1
                        break
                print("***** Step {} *****".format(step))
                samples = self.sampling.GetBatch()
                parameter_values = None
                parameters_normalize = None
                if self.args.only_preloaded_data_exploration:
                    parameter_values = self.evaluate.GetPreEvaluatedParameters(samples, param_list)
                    pruned_parameter_values = self.parameters.GetPrunedSelectedValues(parameter_values, pruned_param_list)
                    parameters_normalize = self.parameters.GetNormalizedParameters(np.array(pruned_parameter_values), pruned_param_list)
                else:
                    parameter_values = self.parameters.GetParameters(samples, param_list)
                    parameters_normalize = self.parameters.GetParameters(samples, pruned_param_list, with_indexing=True, with_normalize=False)
                if self.exploration.IsModelReady():
                    batch_output = self.model.Inference(parameters_normalize)
                else:
                    eval_output = self.evaluate.Run(parameter_values)
                    batch_output = self.extract.Run(parameter_values, param_list, eval_output)
                    (train_acc, val_acc) = self.model.Train(step, pruned_headers, parameters_normalize, batch_output)
                    print("Train accuracy: "+str(train_acc)+" Val accuracy: "+str(val_acc))
                self.WriteExplorationOutput(parameter_values, batch_output)
                step = step + 1
def InitParser(parser):
    parser.add_argument('-config', dest='config', default="config.json")
    parser.add_argument('-only-preloaded-data-exploration', dest='only_preloaded_data_exploration', action="store_true")
    parser.add_argument('-no-run', dest='no_run', action="store_true")
    parser.add_argument('-bounds-no-check', dest='bounds_no_check', action="store_true")
    parser.add_argument('-step-increment', dest='step_inc', default='1')
    parser.add_argument('-step-start', dest='step_start', default='')
    parser.add_argument('-step-end', dest='step_end', default='')
    parser.add_argument('-epochs', dest='epochs', default='-1')
    parser.add_argument('-batch-size', dest='batch_size', default='-1')
    
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
