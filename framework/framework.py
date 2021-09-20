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


from read_config import *
from parameters import *
import numpy as np
from workload_excel import *
from deffe_utils import *
from deffe_thread import *
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
    def __init__(self):
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()

    # Read arguments provided in JSON configuration file
    def ReadArguments(self):
        args = self.parser.parse_args()
        return args

    # Add command line arguments to parser
    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-config", dest="config", default="config.json")
        parser.add_argument(
            "-only-preloaded-data-exploration",
            dest="only_preloaded_data_exploration",
            action="store_true",
        )
        parser.add_argument("-no-run", dest="no_run", action="store_true", help="Dryrun: Do not run!")
        parser.add_argument("-no-train", dest="no_train", action="store_true", help="No training on evaluated metrics")
        parser.add_argument("-bounds-no-check", dest="bounds_no_check", action="store_true")
        parser.add_argument("-step-increment", type=int, dest="step_inc", default=1)
        parser.add_argument("-step-start", type=int, dest="step_start", default=0)
        parser.add_argument("-step-end", type=int, dest="step_end", default=-1)
        parser.add_argument("-epochs", dest="epochs", default="-1", help="Number of epochs set for training")
        parser.add_argument("-batch-size", dest="batch_size", type=int, default=-1, help="Size of batch for sampling, evaluation and extraction")
        parser.add_argument("-evaluate-batch-size", type=int, dest="evaluate_batch_size", default=-1, help="Size of batch for evaluation")
        parser.add_argument("-extract-batch-size", type=int, dest="extract_batch_size", default=-1, help="Size of batch for extraction")
        parser.add_argument("-inference-only", dest="inference_only", action="store_true", help="Use pretrained model for inference")
        parser.add_argument("-input", dest="input", default="")
        parser.add_argument("-output", dest="output", default="")
        parser.add_argument("-model-extract-dir", dest="model_extract_dir", default="")
        parser.add_argument(
            "-model-stats-output", dest="model_stats_output", default="test-output.csv"
        )
        parser.add_argument(
            "-full-exploration", dest="full_exploration", action="store_true"
        )
        parser.add_argument("-fixed-samples", type=int, dest="fixed_samples", default=-1)
        parser.add_argument("-train-test-split", dest="train_test_split", default="")
        parser.add_argument("-validation-split", dest="validation_split", default="")
        parser.add_argument("-init-batch-samples", dest="init_batch_samples", default="100")
        parser.add_argument("-load-train-test", dest="load_train_test", action="store_true")
        parser.add_argument("-hold-evaluated-data", dest="hold_evaluated_data", action="store_true")
        parser.add_argument("-no-slurm", dest="no_slurm", action="store_true", help="No slurm usage")
        parser.add_argument("-sequential", dest="sequential", action="store_true", help="Use sequential mode of deffe evaluation instead of stream mode")
        parser.add_argument("-icp", dest="icp", default="")
        parser.add_argument("-loss", dest="loss", default="")
        return parser

    # Initialize the class objects with default values
    def Initialize(self):
        config = DeffeConfig(self.args.config)
        self.config = config
        self.config_dir = os.path.dirname(self.config.json_file)
        self.init_n_train = int(self.args.init_batch_samples) 
        self.init_n_val = 2 * self.init_n_train
        self.InitializePythonPaths()
        self.predicted_flag = 0
        self.pre_evaluated_flag = 1
        self.evaluate_flag = 2
        self.valid_flag = 1
        self.not_valid_flag = 0
        self.fr_config = self.config.GetFramework()
        self.exploration_table = Workload()
        self.evaluation_table = Workload()
        self.ml_predict_table = Workload()
        self.evaluation_predict_table = Workload()
        self.parameters = Parameters(self.config, self)
        self.model = LoadModule(self, self.config.GetModel().pyscript)
        self.sampling = LoadModule(self, self.config.GetSampling().pyscript)
        #self.exploration = LoadModule(self, self.config.GetExploration().pyscript)
        self.evaluate = LoadModule(self, self.config.GetEvaluate().pyscript)
        self.extract = LoadModule(self, self.config.GetExtract().pyscript)
        self.slurm = LoadModule(self, self.config.GetSlurm().pyscript)
        self.param_data = LoadModule(self, "param_data.py")
        #self.exploration.Initialize()
        self.full_exploration = self.args.full_exploration
        if self.args.inference_only:
            self.full_exploration = True
        self.model_evaluate_flag = False
        self.no_train_flag = self.args.no_train
        self.only_preloaded_data_exploration = \
                            self.args.only_preloaded_data_exploration
        if self.args.input != "":
            self.only_preloaded_data_exploration = True
            self.no_train_flag = True
        if self.only_preloaded_data_exploration:
            if self.args.model_extract_dir != "":
                self.no_train_flag = True
                self.model_evaluate_flag = True

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
        for index, (valid_flag, eval_type, cost_metrics) in enumerate(batch_output):
            param_val = parameter_values[index].tolist()
            cost_metrics = cost_metrics.tolist()
            if eval_type == self.evaluate_flag:
                self.evaluation_table.WriteDataInCSV(param_val + cost_metrics)
                self.evaluation_predict_table.WriteDataInCSV(param_val + cost_metrics)
            elif eval_type == self.predicted_flag:
                self.ml_predict_table.WriteDataInCSV(param_val + cost_metrics)
                self.evaluation_predict_table.WriteDataInCSV(param_val + cost_metrics)

    # Returns true if model is ready
    def IsModelReady(self):
        if self.model.IsModelReady():
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
        self.model.Initialize(self.config.GetCosts(), valid_costs)
        # Initialize the random sampling
        init_n_train = self.init_n_train
        init_n_val = self.init_n_val
        # Preload the data if anything is configured
        if self.only_preloaded_data_exploration:
            n_samples = len(self.param_data.param_data_hash)
            if self.args.model_extract_dir != "" or self.full_exploration:
                train_test_percentage = self.model.GetTrainTestSplit()
                init_n_train = int(n_samples * train_test_percentage)
                init_n_val = n_samples - init_n_train
        self.sampling.Initialize(self.parameters, 
                n_samples, init_n_train, init_n_val
        )

        # Initialize writing of output log files
        hdrs_write_list = [d[0].name for d in param_list] 
        self.config.WriteFile(
            explore_groups.name + "-minmax.json",
            self.parameters.GetMinMaxToJSonData(),
        )
        self.evaluation_table.WriteHeaderInCSV(
            explore_groups.evaluation_table,
            hdrs_write_list + self.config.GetCosts(),
        )
        if not self.no_train_flag:
            self.ml_predict_table.WriteHeaderInCSV(
                explore_groups.ml_predict_table,
                hdrs_write_list + self.config.GetCosts(),
            )
        self.evaluation_predict_table.WriteHeaderInCSV(
            explore_groups.evaluation_predict_table,
            hdrs_write_list + self.config.GetCosts(),
        )
        self.sampling.SetStepInit(0, self.args.step_start,
                self.args.step_end, self.args.step_inc)
        return pruned_headers, param_list, pruned_param_list

    def GetBatchSamples(self, exp_index):
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

    def ExtractParameterValues(self, samples, param_list, pruned_param_list):
        parameter_values = None
        parameters_normalize = None
        # Check if the data point already exist in pre-computed data
        if self.only_preloaded_data_exploration:
            parameter_values = self.param_data.GetPreEvaluatedParameters(
                samples, param_list
            )
            pruned_parameter_values = self.parameters.GetPrunedSelectedValues(
                parameter_values, pruned_param_list
            )
            parameters_normalize = self.parameters.GetNormalizedParameters(
                np.array(pruned_parameter_values), pruned_param_list
            )
        else:
            parameter_values = self.parameters.GetParameters(
                samples, param_list
            )
            parameters_normalize = self.parameters.GetParameters(
                samples,
                pruned_param_list,
                with_indexing=False,
                with_normalize=True,
            )
        return parameter_values, parameters_normalize

    def Inference(self, samples, 
            pruned_headers, parameters_normalize, step
            ):
        self.model.InitializeSamples(
            samples, pruned_headers, 
            parameters_normalize, None, step
        )
        batch_output = self.model.Inference(self.args.output)
        return batch_output

    def Train(self, samples, 
            pruned_headers, parameters_normalize, 
            batch_output, step, threading_model=False):
        self.model.InitializeSamples(
            samples,
            pruned_headers,
            parameters_normalize,
            batch_output,
            step,
        )
        stats_data = self.model.Train(threading_model)
        Log(
            "Stats: (Step, CostI, Epoch, TrainLoss, ValLoss, TrainCount, TestCount): "
            + str(stats_data)
        )
        return stats_data

    def Run(self):
        if self.args.sequential:
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
                parameter_values, parameters_normalize = \
                            self.ExtractParameterValues(samples, 
                                    param_list, pruned_param_list)
                # Check if model is already ready
                if self.IsModelReady() or self.args.inference_only:
                    batch_output =self.Inference(samples, pruned_headers, 
                            parameters_normalize, step)
                else:
                    eval_output = self.evaluate.Run(parameter_values)
                    batch_output = self.extract.Run(
                        parameter_values, param_list, eval_output
                    )
                    if not self.no_train_flag:
                        self.Train(samples, 
                            pruned_headers, parameters_normalize, 
                            batch_output, step)
                    if self.model_evaluate_flag:
                        self.model.InitializeSamples(
                            samples, pruned_headers, 
                            parameters_normalize, batch_output, step
                        )
                        all_files = glob.glob(
                            os.path.join(self.args.model_extract_dir, "*.hdf5")
                        )
                        self.model.EvaluateModel(all_files, self.args.model_stats_output)
                self.WriteExplorationOutput(parameter_values, batch_output)

    # Run the framework
    def RunParallel(self):
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
                if threading_model:
                    while not self.sampling.IsCompleted():
                        samples_with_step = self.GetBatchSamples(exp_index)
                        #LogModule("Generating samples")
                        self.samples_thread.Put('samples', 
                                DeffeThreadData(samples_with_step))
                    self.samples_thread.SendEnd()
                    return True
                else:
                    if not self.sampling.IsCompleted():
                        samples_with_step = self.GetBatchSamples(exp_index)
                        #LogModule("Generating samples")
                        self.samples_thread.Put('samples', 
                                DeffeThreadData(samples_with_step))
                        return False
                    else:
                        self.samples_thread.SendEnd()
                        return True

            def ExtractParamValuesThread(self, exp_index, threading_model=True):
                global_th_end = False
                while True:
                    #LogModule("Inside")
                    (samples_with_step, th_end) = \
                                     self.param_thread.Get('samples').Get()
                    # Check if valid sample, otherwise exit
                    #LogModule(" Received "+str(samples_with_step))
                    global_th_end = th_end
                    if global_th_end:
                        break

                    #LogModule(" Got Data")
                    (step, samples) = samples_with_step
                    parameter_values, parameters_normalize = \
                                self.ExtractParameterValues(samples, 
                                        param_list, pruned_param_list)
                    send_data = {
                        'samples' : DeffeThreadData(samples_with_step),
                        'parameter_values' : DeffeThreadData(parameter_values),
                        'parameters_normalize' : 
                            DeffeThreadData(parameters_normalize),
                    }
                    self.param_thread.PutAll(send_data)
                    if not threading_model:
                        break
                if global_th_end:
                    self.param_thread.SendEnd()

            def InferenceThread(self, threading_model=True):
                global_th_end = False
                while True:
                    #LogModule(" Inside")
                    data_hash = self.inference_thread.GetAll()
                    #LogModule(" Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    #LogModule(" Received "+str(samples_with_step))
                    global_th_end = th_end
                    if global_th_end:
                        break
                    # Check if model is already ready
                    if self.IsModelReady() or self.args.inference_only:
                        #LogModule(" Inferencing now")
                        (step, samples) = samples_with_step
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        batch_output =self.Inference(samples, pruned_headers, 
                                    parameters_normalize, step)
                        data_hash = {
                            'batch_output_inference' : 
                                DeffeThreadData((parameter_values,
                                            batch_output)),
                        }
                        self.inference_thread.PutAll(data_hash)
                    if not threading_model:
                        break
                if global_th_end:
                    self.inference_thread.SendEnd()

            def EvaluateExtractThread(self, threading_model=True):
                global_th_end = False
                while True:
                    #LogModule(" Inside")
                    data_hash = self.evaluate_extract_thread.GetAll()
                    #LogModule(" Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    #LogModule(" Received "+str(samples_with_step))
                    global_th_end = th_end
                    if global_th_end:
                        break
                    # Check if model is already ready
                    if not (self.IsModelReady() or self.args.inference_only):
                        #LogModule(" Started Evaluation "+str(samples_with_step))
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        eval_output = self.evaluate.Run(parameter_values)
                        batch_output = self.extract.Run(
                            parameter_values, param_list, eval_output
                        )
                        data_hash = {
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
                        self.evaluate_extract_thread.PutAll(data_hash)
                    if not threading_model:
                        break
                if global_th_end:
                    self.evaluate_extract_thread.SendEnd()

            def MLTrainThread(self, threading_model=True):
                while True:
                    #LogModule(" Inside")
                    data_hash = self.ml_train_thread.GetAll()
                    #LogModule(" Got Data")
                    (samples_with_step, th_end) = data_hash['samples'].Get()
                    #LogModule(" Received "+str(samples_with_step))
                    if th_end:
                        break
                    # Check if model is already ready
                    if not (self.IsModelReady() or self.args.inference_only):
                        (step, samples) = samples_with_step
                        parameter_values = data_hash['parameter_values'].GetData()
                        parameters_normalize = data_hash['parameters_normalize'].GetData()
                        batch_output = data_hash['batch_output'].GetData()
                        if not self.no_train_flag:
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
                while True:
                    #LogModule(" Inside")
                    while self.write_thread.IsEmpty('batch_output_evaluate') \
                        and self.write_thread.IsEmpty('batch_output_inference'):
                        None
                    #LogModule(" Got Data")
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

            self.samples_thread = DeffeThread(GetSamplesThread, 
                    (self, exp_index, threading_model), True)
            self.param_thread = DeffeThread(
                    ExtractParamValuesThread,
                    (self, exp_index, threading_model), True
                    )
            self.inference_thread = DeffeThread(
                    InferenceThread, (self, threading_model), True)
            self.evaluate_extract_thread = DeffeThread(
                    EvaluateExtractThread, (self, threading_model), True)
            self.ml_train_thread = DeffeThread(
                    MLTrainThread, (self, threading_model), True)
            self.write_thread = DeffeThread(
                    WriteThread, (self, threading_model), True)

            DeffeThread.Connect(self.samples_thread, self.param_thread, 
                    'samples')
            DeffeThread.Connect(self.param_thread, 
                    [self.inference_thread, self.evaluate_extract_thread],
                    ['samples', 'parameter_values', 'parameters_normalize'])
            DeffeThread.Connect(self.evaluate_extract_thread, 
                    self.ml_train_thread,
                    ['samples', 'parameter_values', 
                    'parameters_normalize', 'batch_output'])
            DeffeThread.Connect(self.evaluate_extract_thread, 
                    self.write_thread,
                    ['batch_output_evaluate'])
            DeffeThread.Connect(self.inference_thread, 
                    self.write_thread,
                    ['batch_output_inference'])
            if threading_model:
                self.samples_thread.StartThread()
                self.param_thread.StartThread()
                self.evaluate_extract_thread.StartThread()
                self.inference_thread.StartThread()
                self.ml_train_thread.StartThread()
                self.write_thread.StartThread()

                self.samples_thread.JoinThread()
                self.param_thread.JoinThread()
                self.evaluate_extract_thread.JoinThread()
                self.inference_thread.JoinThread()
                self.ml_train_thread.JoinThread()
                self.write_thread.JoinThread()
            else:
                while True:
                    sampling_status = GetSamplesThread(self, exp_index, threading_model)
                    if sampling_status:
                        break
                    ExtractParamValuesThread(self, exp_index, threading_model)
                    InferenceThread(self, threading_model)
                    EvaluateExtractThread(self, threading_model)
                    MLTrainThread(self, threading_model)
                    WriteThread(self, threading_model)
            
            if self.model_evaluate_flag:
                all_files = glob.glob(
                    os.path.join(self.args.model_extract_dir, "*.hdf5")
                )
                self.model.EvaluateModel(all_files, self.args.model_stats_output)



