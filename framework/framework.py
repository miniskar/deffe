## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
# @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
#         miniskarnr@ornl.gov
#
# Modification:
#              Baseline code
# Date:        Apr, 2020
# **************************************************************************
###
import glob, os, sys
import socket
import pdb
import signal
import numpy as np
import time
import argparse
import shlex
import pandas as pd

def InitializeDeffe():
    framework_path = os.getenv("DEFFE_DIR")
    #print("File:"+__file__)
    if framework_path == None:
        framework_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        os.environ["DEFFE_DIR"] = framework_path
    print("Deffe framework is found in path: "+os.getenv("DEFFE_DIR"))
    sys.path.insert(0, os.getenv("DEFFE_DIR"))
    sys.path.insert(0, os.path.join(framework_path, "utils"))
    sys.path.insert(0, os.path.join(framework_path, "cost_optimizers"))
    sys.path.insert(0, os.path.join(framework_path, "ml_models"))
    sys.path.insert(0, os.path.join(framework_path, "framework"))
    None

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
    def __init__(self, parser=None):
        self.parser = None
        self.args = None
        InitializeDeffe()

    # Read arguments provided in JSON configuration file
    def ReadArguments(self, args_string=None, args=None):
        if args != None:
            self.args = args
        elif args_string != None:
            self.args = self.parser.parse_args(shlex.split(args_string))
        else:
            self.args = self.parser.parse_args()
        from deffe_utils import Log, EnableDebugFlag
        if self.args.debug:
            EnableDebugFlag()
        Log("python3 {} {}".format(__file__, " ".join(sys.argv)))

    # Add command line arguments to parser
    def InitParser(self, parser=None):
        if parser == None:
            parser = argparse.ArgumentParser()
        parser.add_argument("-config", dest="config", default="config.json")
        parser.add_argument(
            "-only-preloaded-data-exploration",
            dest="only_preloaded_data_exploration",
            action="store_true",
        )
        parser.add_argument("-no-run", dest="no_run", action="store_true", help="Dryrun: Do not run!")
        parser.add_argument("-no-train", dest="no_train", action="store_true", help="No training on evaluated metrics")
        parser.add_argument("-validate-module", "-validate-samples", dest="validate_module", action="store_true")
        parser.add_argument("-bounds-no-check", dest="bounds_no_check", action="store_true")
        parser.add_argument("-sampling-method", dest="sampling_method", default='')
        parser.add_argument("-step-increment", type=int, dest="step_inc", default=1)
        parser.add_argument("-step-start", type=int, dest="step_start", default=0)
        parser.add_argument("-step-end", type=int, dest="step_end", default=-1)
        parser.add_argument("-epochs", dest="epochs", default="-1", help="Number of epochs set for training")
        parser.add_argument("-batch-size", dest="batch_size", type=int, default=-1, help="Size of batch for sampling, evaluation and extraction")
        parser.add_argument("-evaluate-batch-size", type=int, dest="evaluate_batch_size", default=-1, help="Size of batch for evaluation")
        parser.add_argument("-extract-batch-size", type=int, dest="extract_batch_size", default=-1, help="Size of batch for extraction")
        parser.add_argument("-mlmodel-batch-size", type=int, dest="mlmodel_batch_size", default=-1, help="Size of batch for ML model")
        parser.add_argument("-evaluate-out-flow", dest="evaluate_out_flow", type=int, default=-1, help="Evaluate samples output flow (by default it is evaluate batch size")
        parser.add_argument("-extract-out-flow", dest="extract_out_flow", type=int, default=-1, help="Extract samples output flow (by default it is evaluate batch size")
        parser.add_argument("-inference-only", dest="inference_only", action="store_true", help="Use pretrained model for inference")
        parser.add_argument("-input", dest="input", default="")
        parser.add_argument("-pre-evaluated-data", dest="pre_evaluated_data", default="")
        parser.add_argument("-output", dest="output", default="")
        parser.add_argument("-model-extract-dir", dest="model_extract_dir", default="")
        parser.add_argument(
            "-model-stats-output", dest="model_stats_output", default="evaluate-model-output.csv"
        )
        parser.add_argument(
            "-full-exploration", dest="full_exploration", action="store_true"
        )
        parser.add_argument("-fixed-samples", type=int, dest="fixed_samples", default=-1, help='Optional fixed set of samples in each batch')
        parser.add_argument("-max-samples", type=int, dest='max_samples', default=1000000, help='Max number of samples to be explored')
        parser.add_argument("-train-test-split", dest="train_test_split", default="")
        parser.add_argument("-validation-split", dest="validation_split", default="")
        parser.add_argument("-init-batch-samples", type=int, dest="init_batch_samples", default=-1)
        parser.add_argument("-load-train-test", dest="load_train_test", action="store_true")
        parser.add_argument("-debug", dest="debug", action="store_true")
        parser.add_argument("-hold-evaluated-data", dest="hold_evaluated_data", action="store_true")
        parser.add_argument("-no-slurm", dest="no_slurm", action="store_true", help="No slurm usage")
        parser.add_argument("-pipeline", dest="pipeline", action="store_true", help="Use pipeline model of deffe instead of sequential execution of modules")
        parser.add_argument("-icp", dest="icp", nargs='*', default=[])
        parser.add_argument("-loss", dest="loss", default="")
        self.parser = parser

    # Initialize the class objects with default values
    def Initialize(self, config_data=None):
        from deffe_utils import LoadModule
        from deffe_utils import LoadPyModule
        from read_config import DeffeConfig
        from workload_excel import Workload
        from parameters import Parameters
        config = DeffeConfig(self.args.config, config_data)
        self.config = config
        self.config_dir = os.path.dirname(self.config.json_file)
        self.init_n_train = self.args.init_batch_samples 
        if self.init_n_train == -1:
            self.init_n_train = self.args.batch_size
        self.init_n_val = 2 * self.init_n_train
        self.InitializePythonPaths()
        self.predicted_flag = 0
        self.pre_evaluated_flag = 1
        self.evaluate_flag = 2
        self.valid_flag = 1
        self.not_valid_flag = 0
        self.fr_config = self.config.GetFramework()
        self.all_table = Workload()
        self.exploration_table = Workload()
        self.evaluation_table = Workload()
        self.ml_predict_table = Workload()
        self.evaluation_predict_table = Workload()
        self.parameters = Parameters(self.config, self)
        self.train_model = LoadPyModule(self.config.GetModel().pyscript, self)
        self.inference_model = LoadPyModule(self.config.GetModel().pyscript, self)
        self.sampling = LoadPyModule(self.config.GetSampling().pyscript, self)
        #self.exploration = LoadPyModule(self.config.GetExploration().pyscript, self)
        self.evaluate = LoadPyModule(self.config.GetEvaluate().pyscript, self)
        self.extract = LoadPyModule(self.config.GetExtract().pyscript, self)
        self.slurm = LoadPyModule(self.config.GetSlurm().pyscript, self, self.config.GetSlurm())
        self.param_data = LoadPyModule("param_data.py", self)
        #self.exploration.Initialize()
        self.full_exploration = self.args.full_exploration
        if self.args.inference_only:
            self.full_exploration = True
        self.train_model_evaluate_flag = False
        self.no_train_flag = self.args.no_train
        self.only_preloaded_data_exploration = \
                            self.args.only_preloaded_data_exploration
        if self.args.input != "":
            self.only_preloaded_data_exploration = True
        if self.args.inference_only:
            self.no_train_flag = True
        if self.only_preloaded_data_exploration:
            if self.args.model_extract_dir != "":
                self.train_model_evaluate_flag = True

    # Initialize the python paths
    def InitializePythonPaths(self):
        python_paths = self.config.GetPythonPaths()
        for path in python_paths:
            if path[0] != "/":
                sys.path.insert(0, os.path.join(os.getcwd(), path))
            else:
                sys.path.insert(0, path)
        # print(sys.path)

    # Log exploration output into file
    def WriteExplorationOutput(self, parameter_values, batch_output):
        from deffe_utils import Log, LogModule, DebugLogModule
        from deffe_utils import GetScriptExecutionTime
        for index, (valid_flag, eval_type, cost_metrics, run_dir) in enumerate(batch_output):
            #DebugLogModule(f"Writing output to parameters {index}")
            param_val = parameter_values[index]
            if type(param_val) != list:
                param_val = param_val.tolist()
            if type(param_val) != list:
                param_val = [param_val]
            #print("Completed1 Writing output to parameters")
            cost_metrics = np.array(cost_metrics).astype(str).tolist()
            #print("Completed2 Writing output to parameters")
            evaluate_time = GetScriptExecutionTime(os.path.join(run_dir, self.config.GetEvaluate().output_log))
            extract_time = GetScriptExecutionTime(os.path.join(run_dir, self.config.GetExtract().output_log))
            self.all_table.WriteDataInCSV(param_val + cost_metrics + [evaluate_time, extract_time, run_dir])
            if eval_type == self.evaluate_flag:
                self.evaluation_table.WriteDataInCSV(param_val + cost_metrics + [evaluate_time, extract_time, run_dir])
                self.evaluation_predict_table.WriteDataInCSV(param_val + cost_metrics + [evaluate_time, extract_time, run_dir])
            elif eval_type == self.predicted_flag:
                self.ml_predict_table.WriteDataInCSV(param_val + cost_metrics + [evaluate_time, extract_time, run_dir])
                self.evaluation_predict_table.WriteDataInCSV(param_val + cost_metrics + [evaluate_time, extract_time, run_dir])

    # Returns true if model is ready
    def IsModelReady(self):
        if self.train_model.IsModelReady():
            return True
        return False

    def InitializeModulesForExploration(self, exp_index, explore_groups):
        # extract the parameters and cost metrics in that exploration list
        (param_list, pruned_param_list, n_samples) = \
                self.parameters.Initialize(
                        explore_groups.groups
                )
        headers = self.parameters.GetHeaders(param_list)
        pruned_headers = self.parameters.GetHeaders(pruned_param_list)
        load_data_file = explore_groups.pre_evaluated_data
        valid_costs = explore_groups.valid_costs
        if self.args.input != "":
            load_data_file = self.args.input
        if self.args.pre_evaluated_data != "":
            load_data_file = self.args.pre_evaluated_data
        # Initialize the preloaded data, evaluate, extract and ML models
        self.param_data.Initialize(
            param_list, pruned_param_list, 
            self.config.GetCosts(), load_data_file
        )
        self.evaluate.Initialize(
            param_list, pruned_param_list, self.param_data
        )
        self.extract.Initialize(param_list, 
                self.config.GetCosts(), self.param_data
        )
        self.train_model.Initialize(self.config.GetCosts(), valid_costs)
        self.inference_model.Initialize(self.config.GetCosts(), valid_costs)
        # Initialize the random sampling
        init_n_train = self.init_n_train
        init_n_val = self.init_n_val
        #TODO
        if load_data_file != '':
            preload_samples = self.param_data.GetEncodedSamples()
            (batch_output, cost_names, sel_samples) = self.param_data.GetCostData()
            parameter_values = self.parameters.GetParameters(
                preload_samples, param_list
            )
            parameters_normalize = self.parameters.GetParameters(
                preload_samples,
                pruned_param_list,
                with_indexing=False,
                with_normalize=True,
            )
            self.Train(preload_samples, 
                pruned_headers, cost_names, parameters_normalize, 
                batch_output, 0)
            None
        # Preload the data if anything is configured
        if self.only_preloaded_data_exploration:
            n_samples = len(self.param_data.param_data_hash)
            if self.args.model_extract_dir != "" or self.full_exploration:
                train_val_split = self.train_model.GetTrainValSplit()
                init_n_train = int(n_samples * (1.0-train_val_split))
                init_n_val = n_samples - init_n_train
        self.sampling.Initialize(self.parameters, n_samples,
                init_n_train, init_n_val, True, 
                self.train_model.GetTrainValSplit(), self.full_exploration
        )

        # Initialize writing of output log files
        hdrs_write_list = [d[0].name for d in param_list] 
        self.config.WriteFile(
            explore_groups.name + "-minmax.json",
            self.parameters.GetMinMaxToJSonData(),
        )
        self.all_table.WriteHeaderInCSV(
            explore_groups.all_table,
            hdrs_write_list + self.config.GetCosts()+["EvaluateTime","ExtractTime", "RunDir"],
        )
        self.evaluation_table.WriteHeaderInCSV(
            explore_groups.evaluation_table,
            hdrs_write_list + self.config.GetCosts()+["EvaluateTime","ExtractTime", "RunDir"],
        )
        if not self.no_train_flag or self.args.inference_only:
            self.ml_predict_table.WriteHeaderInCSV(
                explore_groups.ml_predict_table,
                hdrs_write_list + self.config.GetCosts()+["EvaluateTime","ExtractTime", "RunDir"],
            )
        self.evaluation_predict_table.WriteHeaderInCSV(
            explore_groups.evaluation_predict_table,
            hdrs_write_list + self.config.GetCosts()+["EvaluateTime","ExtractTime", "RunDir"],
        )
        self.sampling.SetStepInit(0, self.args.step_start,
                self.args.step_end, self.args.step_inc)
        return pruned_headers, param_list, pruned_param_list

    def GetBatchSamples(self, exp_index):
        from deffe_utils import Log
        samples = self.sampling.GetNewBatch()
        step = self.sampling.GetCurrentStep()
        Log("***** Exploration:{}/{} Step {} Current "
                "Samples:{} Samples:{}/{} "
                "*****".format(
                    exp_index+1,
                    self.total_explorations,
                    step,
                    len(samples),
                    len(self.sampling.GetBatch()), 
                    self.parameters.total_permutations))
        return (step, samples)

    def ExtractCostValues(self, samples):
        (cost_data, cost_pd, sel_samples) = self.param_data.GetCostData(samples)
        return cost_data

    def ExtractParameterValues(self, samples, param_list, pruned_param_list):
        from deffe_utils import Log, LogModule, DebugLogModule
        parameter_values = None
        pruned_parameter_values = None
        parameters_normalize = None
        # Check if the data point already exist in pre-computed data
        if self.only_preloaded_data_exploration:
            DebugLogModule("Loading pre evaluated parameters")
            parameter_values = self.param_data.GetPreEvaluatedParameters(
                samples, param_list
            )
            DebugLogModule("Get pruned selected values")
            pruned_parameter_values = self.parameters.GetPrunedSelectedValues(
                parameter_values, pruned_param_list
            )
            DebugLogModule("Get normalized parameters")
            parameters_normalize = self.parameters.GetNormalizedParameters(
                np.array(pruned_parameter_values), pruned_param_list
            )
            DebugLogModule("Completed")
        else:
            DebugLogModule("Get sample parameter values")
            parameter_values = self.parameters.GetParameters(
                samples, param_list
            )
            DebugLogModule("Get pruned selected values")
            pruned_parameter_values = self.parameters.GetPrunedSelectedValues(
                parameter_values, pruned_param_list
            )
            DebugLogModule("Get normalized parameters")
            parameters_normalize = self.parameters.GetParameters(
                samples,
                pruned_param_list,
                with_indexing=False,
                with_normalize=True,
            )
        return parameter_values, pruned_parameter_values, parameters_normalize

    def WritePredictionsToFile(self, model, headers, cost_names, params, cost_data, predictions, outfile):
        from deffe_utils import Log
        Log("Writing output file:" + outfile)
        out_data_hash = {}
        params_tr = params.transpose()
        cost_data_tr = cost_data.transpose()
        predictions_tr = predictions.transpose()
        for index, hdr in enumerate(headers):
            out_data_hash[hdr] = params_tr[index].tolist()
            print(f"Index:{index} Param:{hdr} Len:{len(params_tr[index])}")
        for index, cost in enumerate(cost_names):
            if model.IsValidCost(cost):
                s_true_cost = np.array([])
                s_pred_cost = predictions_tr[index]
                if index < cost_data_tr.shape[0]:
                    s_true_cost = cost_data_tr[index].astype(float)
                    out_data_hash[cost] = s_true_cost.tolist()
                out_data_hash["predicted-"+cost] = s_pred_cost.tolist()
                if index < cost_data_tr.shape[0]:
                    error = np.abs(s_true_cost - s_pred_cost)
                    error_percent = error / s_true_cost
                    out_data_hash["error-"+cost] = error.tolist()
                    out_data_hash["error-percent-"+cost] = error_percent.tolist()
                    print("Index:{} Cost:{} Error: {} Length:{}".format(index, cost, np.mean(error_percent), len(error_percent)))
        df = pd.DataFrame(out_data_hash)
        df.to_csv(outfile, index=False, sep=",", encoding="utf-8")
        return None

    def GetPredictedCost(self, samples, step=0, cost_names=[]):
        pruned_param_list = self.parameters.selected_pruned_params
        parameter_values, pruned_parameter_values, parameters_normalize = \
                    self.ExtractParameterValues(samples, 
                            self.param_data.param_list, 
                            self.param_data.pruned_param_list)
        pruned_headers = self.parameters.GetHeaders(pruned_param_list)
        # Check if model is already ready
        if len(cost_names) == 0:
            cost_names = self.config.GetCosts()
        cost_data =self.Inference(samples, 
                pruned_headers, cost_names, parameter_values,
                parameters_normalize, step)
        cost_hdr_hash = {cost:index for index, cost in enumerate(self.config.GetCosts())}
        out_cost_array = []
        for cost in cost_names:
            if cost in cost_hdr_hash:
                out_cost_array.append(cost_data[cost_hdr_hash[cost]])
            else:
                out_cost_array.append(None)
        return (pruned_headers, cost_names, pruned_parameter_values, out_cost_array)

    def Inference(self, samples, pruned_headers, cost_names,
            parameter_values, parameters_normalize, step,
            cost_data=None, eval_format=False,
            expand_list=False,
            preload_cost_checkpoints=True, outfile=None):
        self.inference_model.InitializeSamples(
            samples, pruned_headers, cost_names,
            parameters_normalize, cost_data, step, 
            expand_list,
            preload_cost_checkpoints
        )
        batch_output = self.inference_model.Inference()
        if outfile != None:
            valid_indexes = self.inference_model.params_valid_indexes
            params = parameter_values
            if type(params) == list:
                params = np.array(params)
            params = params[valid_indexes,]
            cost_data = self.inference_model.cost_output
            pred_data = np.array(batch_output)
            cost_names = self.inference_model.cost_names
            self.WritePredictionsToFile(self.inference_model, pruned_headers, cost_names, params, cost_data, pred_data, outfile)
        if not eval_format:
            return batch_output
        cost = []
        for output in batch_output:
            cost.append(
                (self.valid_flag, self.predicted_flag, output, '')
            )
        return cost 

    def Train(self, samples, 
            pruned_headers, cost_names, parameters_normalize, 
            batch_output, step, threading_model=False):
        from deffe_utils import Log
        self.train_model.InitializeSamples(
            samples,
            pruned_headers,
            cost_names,
            parameters_normalize,
            batch_output,
            step,
        )
        #pdb.set_trace()
        stats_data = self.train_model.Train(threading_model)
        Log(
            "Stats: (Step, CostI, Epoch, TrainLoss, ValLoss, TrainCount, TestCount): "
            + str(stats_data)
        )
        return stats_data

    def Run(self):
        if not self.args.pipeline:
            self.RunSequential()
        else:
            self.RunParallel()

    # Run the framework
    def RunSequential(self):
        if not os.path.exists(self.fr_config.run_directory):
            os.makedirs(self.fr_config.run_directory)
        # Iterate through multiple explorations list as per the configuration
        exploration_list = self.config.GetExploration().exploration_list
        self.total_explorations = len(exploration_list)
        for exp_index, explore_groups in enumerate(exploration_list):
            pruned_headers, param_list, pruned_param_list = \
                self.InitializeModulesForExploration(exp_index,
                    explore_groups)
            # Iterate the exploration until it is completed
            while not self.sampling.IsCompleted():
                (step, samples) = self.GetBatchSamples(exp_index)
                parameter_values, pruned_parameter_values, parameters_normalize = \
                            self.ExtractParameterValues(samples, 
                                    param_list, pruned_param_list)
                # Check if model is already ready
                if self.IsModelReady() or self.args.inference_only:
                    # Rethink about this
                    cost_data = self.ExtractCostValues(samples)
                    batch_output =self.Inference(samples, 
                            pruned_headers, self.config.GetCosts(), 
                            parameter_values,
                            parameters_normalize, step, 
                            cost_data, eval_format=True, outfile=self.args.output)
                else:
                    eval_output = self.evaluate.Run(parameter_values)
                    batch_output = self.extract.Run(
                        parameter_values, param_list, eval_output
                    )
                    if not self.no_train_flag:
                        self.Train(samples, 
                            pruned_headers, self.config.GetCosts(), 
                            parameters_normalize, 
                            batch_output, step)
                    if self.train_model_evaluate_flag:
                        self.train_model.InitializeSamples(
                            samples, pruned_headers, self.config.GetCosts(),  
                            parameters_normalize, batch_output, step
                        )
                        all_files = glob.glob(
                            os.path.join(self.args.model_extract_dir, "*.hdf5")
                        )
                        self.train_model.EvaluateModel(all_files, self.args.model_stats_output)
                self.WriteExplorationOutput(parameter_values, batch_output)

    # Run the framework
    def RunParallel(self):
        from deffe_thread import DeffeThread, DeffeThreadData
        from deffe_utils import Log, DebugLogModule
        threading_model = True
        if not os.path.exists(self.fr_config.run_directory):
            os.makedirs(self.fr_config.run_directory)
        # Iterate through multiple explorations list as per the configuration
        exploration_list = self.config.GetExploration().exploration_list
        self.total_explorations = len(exploration_list)
        for exp_index, explore_groups in enumerate(exploration_list):
            pruned_headers, param_list, pruned_param_list = \
                self.InitializeModulesForExploration(exp_index,
                    explore_groups)
            def GetSamplesThread(self, exp_index, threading_model=True):
                # OUT Ports: samples
                if threading_model:
                    while not self.sampling.IsCompleted():
                        samples_with_step = self.GetBatchSamples(exp_index)
                        DebugLogModule("Generating samples")
                        self.samples_thread.Put('samples', 
                                DeffeThreadData(samples_with_step))
                        DebugLogModule("Sent samples:"+str(len(samples_with_step[1])))
                    DebugLogModule("Samples: Sending last sample end")
                    self.samples_thread.SendEnd()
                    return True
                else:
                    if not self.sampling.IsCompleted():
                        samples_with_step = self.GetBatchSamples(exp_index)
                        #DebugLogModule("Generating samples")
                        DebugLogModule("Sending data:"+str(len(samples_with_step[1])))
                        self.samples_thread.Put('samples', 
                                DeffeThreadData(samples_with_step))
                        DebugLogModule("Sent data:"+str(len(samples_with_step[1])))
                        return False
                    else:
                        DebugLogModule("Samples: Sending last sample end")
                        self.samples_thread.SendEnd()
                        DebugLogModule("Samples: Sent last sample end")
                        return True

            def ExtractParamValuesThread(self, exp_index, threading_model=True):
                global_th_end = False
                # IN Ports: samples
                # OUT Ports: samples, parameter_values, parameters_normalize
                while True:
                    DebugLogModule("Inside Param Threads")
                    DebugLogModule("Trying to fetch Param Threads")
                    (samples_with_step, th_end) = \
                                     self.param_thread.Get('samples').Get()
                    # Check if valid sample, otherwise exit
                    global_th_end = th_end
                    if global_th_end:
                        DebugLogModule("Received thread end")
                        break
                    DebugLogModule("SSSSS:{}".format(samples_with_step))
                    DebugLogModule("Received "+str(len(samples_with_step[1])))
                    DebugLogModule("Got Data")
                    (step, samples) = samples_with_step
                    DebugLogModule("Extracting parameter values data")
                    parameter_values, pruned_parameter_values, parameters_normalize = \
                                self.ExtractParameterValues(samples, 
                                        param_list, pruned_param_list)
                    DebugLogModule("Extracted parameter values data")
                    send_data = {
                        'samples' : DeffeThreadData(samples_with_step),
                        'parameter_values' : DeffeThreadData(parameter_values),
                        'parameters_normalize' : 
                            DeffeThreadData(parameters_normalize),
                    }
                    DebugLogModule("Sending data:"+str(len(parameter_values)))
                    self.param_thread.PutAll(send_data)
                    DebugLogModule("Data sent")
                    if not threading_model:
                        break
                if global_th_end:
                    DebugLogModule("ExtractParams: Sending last sample end")
                    self.param_thread.SendEnd()

            def MLInferenceThread(self, threading_model=True):
                global_th_end = False
                # IN Ports: samples
                # IN Ports (Cond): parameter_values, parameters_normalize
                # OUT Ports: batch_output_inference
                while True:
                    DebugLogModule("Inside")
                    data_hash = self.inference_thread.GetAll()
                    DebugLogModule("Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    global_th_end = th_end
                    if global_th_end:
                        DebugLogModule("Received thread end")
                        break
                    DebugLogModule("Received "+str(len(samples_with_step[1])))
                    # Check if model is already ready
                    if self.IsModelReady() or self.args.inference_only:
                        DebugLogModule("Inferencing now")
                        (step, samples) = samples_with_step
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        batch_output =self.Inference(samples, 
                                pruned_headers, 
                                self.config.GetCosts(), parameter_values, 
                                parameters_normalize, step, 
                                eval_format=True)
                        data_hash = {
                            'batch_output_inference' : 
                                DeffeThreadData((parameter_values,
                                            batch_output)),
                        }
                        DebugLogModule("Sending data:"+str(len(parameter_values)))
                        self.inference_thread.PutAll(data_hash)
                        DebugLogModule("Sent data:"+str(len(parameter_values)))
                    if not threading_model:
                        break
                if global_th_end:
                    DebugLogModule("Inference: Sending last sample end")
                    self.inference_thread.SendEnd()

            def EvaluateThread(self, threading_model=True):
                from threading import Lock
                global_th_end = False
                eval_output_list = []
                eval_stats = [0, 0, False, False] # Submitted, Completed, Th-End, End-packet-Flag
                def pushDataToQueue(self, eval_output_list, in_data_hash):
                    np_list = np.array(eval_output_list).transpose().tolist()
                    samples_with_step = in_data_hash['samples']
                    samples_step = samples_with_step[0]
                    samples = np.array(samples_with_step[1])[np_list[0]].tolist()
                    samples_with_step = (samples_step, samples)
                    parameter_values = np.array(in_data_hash['parameter_values'])[np_list[0]].tolist()
                    parameters_normalize = np.array(in_data_hash['parameters_normalize'])[np_list[0]].tolist()
                    out_data_hash = {
                        'samples' : DeffeThreadData(samples_with_step),
                        'parameter_values' : 
                            DeffeThreadData(parameter_values),
                        'parameters_normalize' : 
                            DeffeThreadData(parameters_normalize),
                        'eval_output' : 
                            DeffeThreadData(np_list[1]),
                    }
                    #print("Prepared data to send to extract")
                    DebugLogModule("Sending data:"+str(len(parameter_values)))
                    self.evaluate_thread.PutAll(out_data_hash)
                    DebugLogModule("Sent data:"+str(len(parameter_values)))
                    #print("Completed putall data to send to extract")
                    eval_output_list.clear()
                    
                def callbackEvaluate(self, update_mutex, eval_output_list, eval_stats, in_data_hash, index, eval_output):
                    update_mutex.acquire(1)
                    #print("Acquiring the lock")
                    eval_output_list.append([index, eval_output])
                    eval_stats[1] += 1
                    th_end = eval_stats[2]
                    eval_out_flow = self.evaluate.GetOutputFlow()
                    is_it_last_sample = th_end and eval_stats[0] == eval_stats[1]
                    #print("Is last sample:"+str(is_it_last_sample)+" th.end:"+str(th_end))
                    #print(f"Evaluate index:{index} completed size:"+str(len(eval_output_list))+" Eval stats:"+str(eval_stats))
                    if len(eval_output_list) >= eval_out_flow or is_it_last_sample:
                        #print("Send of batch:"+str(len(eval_output_list)))
                        pushDataToQueue(self, eval_output_list, in_data_hash)
                        #print("Successfully Sent of batch:"+str(len(eval_output_list)))
                        if is_it_last_sample and not eval_stats[3]:
                            #print("Ealuate Sending last sample end")
                            DebugLogModule("Sending End data")
                            self.evaluate_thread.SendEnd()
                            eval_stats[3] = True
                        #print("Completed Send of batch:"+str(len(eval_output_list)))
                    #print("Releasing the lock")
                    update_mutex.release()
                # IN Ports: samples
                # IN Ports (Cond): parameter_values, parameters_normalize
                # OUT Ports: samples, parameter_values, parameters_normalize, eval_output
                #send_mutex = Lock()
                update_mutex = Lock()
                while True:
                    DebugLogModule("Inside")
                    data_hash = self.evaluate_thread.GetAll()
                    DebugLogModule("Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    eval_stats[2] = th_end
                    global_th_end = th_end
                    if global_th_end:
                        DebugLogModule("Received thread end")
                        update_mutex.acquire(1)
                        if eval_stats[0] == eval_stats[1] and not eval_stats[3]:
                            self.evaluate_thread.SendEnd()
                        update_mutex.release()
                        break
                    DebugLogModule("Received "+str(len(samples_with_step[1])))
                    # Check if model is already ready
                    if not (self.IsModelReady() or self.args.inference_only):
                        DebugLogModule("Started Evaluation "+str(len(samples_with_step)))
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        in_data_hash={
                            'th_end' : th_end,
                            'samples' : samples_with_step,
                            'parameter_values' : parameter_values,
                            'parameters_normalize' : parameters_normalize
                        }
                        eval_stats[0] += len(parameter_values)
                        eval_output = self.evaluate.Run(parameter_values, 
                                (self, callbackEvaluate, update_mutex, 
                                     eval_output_list, eval_stats, in_data_hash), 
                                use_global_thread_queue=True)
                        #data_hash = {
                        #    'samples' : DeffeThreadData(samples_with_step),
                        #    'parameter_values' : 
                        #        DeffeThreadData(parameter_values),
                        #    'parameters_normalize' : 
                        #        DeffeThreadData(parameters_normalize),
                        #    'eval_output' : 
                        #        DeffeThreadData(eval_output),
                        #}
                        #self.evaluate_thread.PutAll(data_hash)
                    if not threading_model:
                        break
                #if global_th_end:
                    #self.evaluate_thread.SendEnd()

            def ExtractResultsThread(self, threading_model=True):
                global_th_end = False
                # IN Ports: samples
                # IN Ports (Cond): parameter_values, parameters_normalize, eval_output
                # OUT Ports: samples, parameter_values, parameters_normalize, 
                #            batch_output, batch_output_evaluate, batch_output_inference
                while True:
                    DebugLogModule("Inside")
                    data_hash = self.extract_thread.GetAll()
                    DebugLogModule("Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    global_th_end = th_end
                    if global_th_end:
                        DebugLogModule("Received thread end")
                        break
                    if samples_with_step == None:
                        break
                    DebugLogModule("Received "+str(len(samples_with_step)))
                    # Check if model is already ready
                    if not (self.IsModelReady() or self.args.inference_only):
                        DebugLogModule("Started Evaluation "+str(len(samples_with_step)))
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        eval_output = data_hash['eval_output'].GetData()
                        batch_output = self.extract.Run(
                            parameter_values, param_list, eval_output
                        )
                        out_data_hash = {
                            'samples' : DeffeThreadData(samples_with_step),
                            'parameter_values' : 
                                DeffeThreadData(parameter_values),
                            'parameters_normalize' : 
                                DeffeThreadData(parameters_normalize),
                            'batch_output' : 
                                DeffeThreadData(batch_output),
                            'batch_output_evaluate' : 
                                DeffeThreadData((parameter_values,
                                            batch_output)),
                        }
                        DebugLogModule("Sending data")
                        self.extract_thread.PutAll(out_data_hash)
                        DebugLogModule("Sent data")
                    if not threading_model:
                        break
                if global_th_end:
                    DebugLogModule("Extract: Inference last sample end")
                    self.extract_thread.SendEnd()

            def MLTrainThread(self, threading_model=True):
                # IN Ports: samples
                # IN Ports (Cond): parameter_values, parameters_normalize, batch_output
                # OUT Ports: NONE
                while True:
                    DebugLogModule("Inside")
                    data_hash = self.ml_train_thread.GetAll()
                    DebugLogModule("Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    if th_end:
                        DebugLogModule("Received thread end")
                        break
                    DebugLogModule("Received "+str(len(samples_with_step[1])))
                    # Check if model is already ready
                    if not (self.IsModelReady() or self.args.inference_only):
                        (step, samples) = samples_with_step
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        batch_output = data_hash['batch_output'].GetData()
                        if not self.no_train_flag:
                            DebugLogModule("Started training")
                            self.Train(samples, 
                                pruned_headers, parameters_normalize, 
                                batch_output, step, threading_model)
                    if not threading_model:
                        break

            def WriteThread(self, threading_model=True):
                def Process(self, data):
                    (parameter_values, batch_output) = data
                    self.WriteExplorationOutput(parameter_values, batch_output)

                global_eval_th = False
                global_inf_th = False
                # OUT Ports: batch_output_evaluate, batch_output_inference
                while True:
                    DebugLogModule("Inside")
                    while self.write_thread.IsEmpty('batch_output_evaluate') \
                        and self.write_thread.IsEmpty('batch_output_inference'):
                        None
                    DebugLogModule("Got Data")
                    if not self.write_thread.IsEmpty('batch_output_evaluate'):
                        (eval_data, eval_th)  = \
                            self.write_thread.Get('batch_output_evaluate').Get()
                        if not eval_th:
                            Process(self, eval_data)
                        else:
                            global_eval_th = eval_th
                    if not self.write_thread.IsEmpty('batch_output_inference'):
                        (inf_data, inf_th)  = \
                            self.write_thread.Get('batch_output_inference').Get()
                        if not inf_th:
                            Process(self, inf_data)
                        else:
                            global_inf_th = inf_th
                    if global_eval_th and global_inf_th:
                        if threading_model:
                            break
                        return
                    if not threading_model:
                        break

            print("********************************************")
            self.samples_thread = DeffeThread(GetSamplesThread, 
                    (self, exp_index, threading_model), True)
            self.param_thread = DeffeThread(
                    ExtractParamValuesThread,
                    (self, exp_index, threading_model), True
                    )
            self.inference_thread = DeffeThread(
                    MLInferenceThread, (self, threading_model), True)
            self.evaluate_thread = DeffeThread(
                    EvaluateThread, (self, threading_model), True)
            self.extract_thread = DeffeThread(
                    ExtractResultsThread, (self, threading_model), True)
            self.ml_train_thread = DeffeThread(
                    MLTrainThread, (self, threading_model), True)
            self.write_thread = DeffeThread(
                    WriteThread, (self, threading_model), True)

            DeffeThread.Connect(self.samples_thread, self.param_thread, 
                    'samples')
            DeffeThread.Connect(self.param_thread, 
                    [self.inference_thread, self.evaluate_thread],
                    ['samples', 'parameter_values', 'parameters_normalize'])
            DeffeThread.Connect(self.evaluate_thread, 
                    [self.extract_thread],
                    ['samples', 'parameter_values', 'parameters_normalize', 'eval_output'])
            DeffeThread.Connect(self.extract_thread, 
                    self.ml_train_thread,
                    ['samples', 'parameter_values', 
                    'parameters_normalize', 'batch_output'])
            DeffeThread.Connect(self.extract_thread, 
                    self.write_thread,
                    ['batch_output_evaluate'])
            DeffeThread.Connect(self.inference_thread, 
                    self.write_thread,
                    ['batch_output_inference'])
            if threading_model:
                self.samples_thread.StartThread()
                self.param_thread.StartThread()
                self.evaluate_thread.StartThread()
                self.extract_thread.StartThread()
                self.inference_thread.StartThread()
                self.ml_train_thread.StartThread()
                self.write_thread.StartThread()

                self.samples_thread.JoinThread()
                self.param_thread.JoinThread()
                self.evaluate_thread.JoinThread()
                self.extract_thread.JoinThread()
                self.inference_thread.JoinThread()
                self.ml_train_thread.JoinThread()
                self.write_thread.JoinThread()
            else:
                while True:
                    sampling_status = GetSamplesThread(self, exp_index, threading_model)
                    if sampling_status:
                        break
                    ExtractParamValuesThread(self, exp_index, threading_model)
                    MLInferenceThread(self, threading_model)
                    EvaluateThread(self, threading_model)
                    ExtractResultsThread(self, threading_model)
                    MLTrainThread(self, threading_model)
                    WriteThread(self, threading_model)
            
            if self.train_model_evaluate_flag:
                all_files = glob.glob(
                    os.path.join(self.args.model_extract_dir, "*.hdf5")
                )
                self.train_model.EvaluateModel(all_files, self.args.model_stats_output)



