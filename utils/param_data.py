import os
import pdb
import re
import sys
import numpy as np
import pandas as pd
import argparse
import shlex
from deffe_utils import *

class ParamData:
    def __init__(self, framework):
        self.framework = framework
        self.param_data_hash = {}
        self.param_extract_indexes = []
        self.unused_params_values = []
        self.cost_data = np.array([])
        self.param_data_sample_indexes = np.array([])
        self.cost_data_pd = pd.DataFrame()

    def GetEncodedSamples(self):
        return self.param_data_sample_indexes

    def GetCostDataPandas(self, indexes=None):
        if indexes == None:
            return self.cost_data_pd
        return self.cost_data_pd.loc[indexes]

    def GetCostData(self, samples=None):
        indexes = list(range(self.cost_data.shape[0]))
        if samples != None:
            indexes = np.where(np.isin(self.param_data_sample_indexes, samples))
        sel_samples = self.param_data_sample_indexes[indexes]
        cost_pd = self.cost_data_pd.loc[indexes]
        return ([(self.framework.valid_flag, self.framework.pre_evaluated_flag, np.array(x), '') for x in self.cost_data[indexes].tolist()], cost_pd.columns, sel_samples)

    # Initialize method should be called for every new instance of new batch of samples.
    # Parameters to be passed: Parameters list, Pruned parameters list, Cost metrics names, and also
    # if any preload_file (Pre-Evaluated results)
    def Initialize(self, param_list, pruned_param_list, 
            cost_list, preload_file):
        self.param_data_hash = {}
        self.param_extract_indexes = []
        self.unused_params_values = []
        self.param_list = param_list
        self.pruned_param_list = pruned_param_list
        self.pruned_list_indexes = []
        for (param, param_values, pindex, permutation_index) in self.pruned_param_list:
            self.pruned_list_indexes.append(pindex)
        self.cost_list = cost_list
        param_hdrs = []
        param_hash = {}
        cost_hash = {}
        for index, cost in enumerate(self.cost_list):
            cost_hash[cost.lower()] = index
            cost_hash[cost] = index
        for pdata in self.param_list:
            (param, param_values, pindex, permutation_index) = pdata
            param_hdrs.append(param.name.lower())
            param_hash[param.name.lower()] = pdata
            param_hash[param.map.lower()] = pdata
        self.param_hash = param_hash
        for (param, param_values, pindex, permutation_index) in self.param_list:
            self.param_extract_indexes.append(pindex)
        if preload_file == None:
            return
        if not os.path.exists(preload_file):
            preload_file = os.path.join(self.framework.config_dir, preload_file)
        if not os.path.exists(preload_file):
            return
        print(f"Loading preevaluated data: {preload_file}")
        pd_data = pd.read_csv(
            preload_file, dtype="str", delimiter=r"\s*,\s*", engine="python"
        )
        tmp_lower_columns = pd_data.columns.str.lower()
        for index, hdr in enumerate(cost_list):
            if hdr.lower() not in tmp_lower_columns:
                print(f"Couldn't find data for {hdr} in preloaded data file")
                pd_data[hdr.lower()] = np.nan
        print(f"Size of preevaluated data: {pd_data.shape}")
        np_data = pd_data.values.astype("str")
        np_hdrs = np.char.lower(np.array(list(pd_data.columns)).astype("str"))
        preload_data = np_data[0:]
        trans_data = preload_data.transpose()
        np_hdr_index_hash = { hdr: index for index, hdr in enumerate(np_hdrs) }
        self.np_param_valid_indexes = []
        self.np_param_hdrs = []
        self.np_cost_valid_indexes = []
        self.np_cost_hdrs = []
        #pdb.set_trace()
        for index, orig_hdr in enumerate(param_hdrs):
            hdr = orig_hdr.lower()
            pindex = np_hdr_index_hash[hdr]
            if hdr in param_hash:
                self.np_param_valid_indexes.append(pindex)
                self.np_param_hdrs.append(hdr)
                param = self.param_hash[hdr][0]
                param_values = list(tuple(trans_data[pindex]))
                is_numbers = self.framework.parameters.IsParameterNumber(param_values)
                if is_numbers:
                    minp = np.min(trans_data[pindex].astype("float"))
                    maxp = np.max(trans_data[pindex].astype("float"))
                    # print("MinP: "+str(minp)+" maxP:"+str(maxp)+" name:"+param.map)
                    self.framework.parameters.UpdateMinMaxRange(param, minp, maxp)
        for index, orig_hdr in enumerate(cost_list):
            hdr = orig_hdr.lower()
            pindex = np_hdr_index_hash[hdr]
            if hdr in cost_hash:
                self.np_cost_hdrs.append(hdr)
                self.np_cost_valid_indexes.append(pindex)
        # self.GetValidPreloadedData(trans_data)
        #pdb.set_trace()
        param_data = trans_data[self.np_param_valid_indexes,].transpose()
        cost_data = trans_data[self.np_cost_valid_indexes,].transpose()
        print("Loaded data items:" + str(param_data.shape[0]))
        #pdb.set_trace()
        valid_indexes = []
        for index in range(len(param_data)):
            tp_data = tuple(param_data[index])
            if tp_data not in self.param_data_hash:
                self.param_data_hash[tp_data] = cost_data[index]
                valid_indexes.append(index)
        self.param_hdrs = self.np_param_hdrs
        self.param_data = param_data[
            valid_indexes,
        ]
        def GetRecHash(rec):
            #print(rec)
            sel_param_hash = { 
                k:rec[index]
                      for index, k in enumerate(self.param_hdrs) 
                }
            return sel_param_hash
        self.param_data_sample_indexes = np.array([
            self.framework.parameters.EncodePermutation( \
                GetRecHash(rec) \
            ) \
            for rec in self.param_data
            ]).astype("int")
        #print(self.param_data_sample_indexes)
        self.cost_data = cost_data[
            valid_indexes,
        ]
        self.param_data_pd = pd.DataFrame(self.param_data, columns=self.param_hdrs)
        self.cost_data_pd = pd.DataFrame(self.cost_data, columns=self.np_cost_hdrs)
        print("Valid data items:" + str(self.param_data.shape[0]))
        np_param_hdrs_hash = {}
        for index, hdr in enumerate(self.np_param_hdrs):
            np_param_hdrs_hash[hdr] = index
        unused_params_list = []
        count = 0
        for pdata in self.param_list:
            (param, param_values, pindex, permutation_index) = pdata
            if len(param_values) > 1:
                if param.name in np_param_hdrs_hash:
                    count = count + 1
                elif param.map in np_param_hdrs_hash:
                    count = count + 1
                else:
                    unused_params_list.append(param.name)
                    self.unused_params_values.append(param_values[0])
            else:
                unused_params_list.append(param.name)
                self.unused_params_values.append(param_values[0])
        #pdb.set_trace()
        #if count != len(self.np_param_hdrs):
        #    LogError("Parameters identidied in file:{} are {}, "
        #        "which is not matching with valid count parameters:{}".
        #        format(preload_file, len(self.np_param_hdrs), count))
        #    return
        self.rev_param_list = self.np_param_hdrs + unused_params_list
        #pdb.set_trace()
        self.param_extract_indexes = [index for index in range(len(self.np_param_hdrs))]
        self.rev_param_extract_indexes = [
            index for index in range(len(self.param_list))
        ]
        for index, hdr in enumerate(self.rev_param_list):
            if hdr in param_hash:
                self.rev_param_extract_indexes[param_hash[hdr][2]] = index
        for (param, param_values, pindex, permutation_index) in self.param_list:
            if len(param_values) > 1:
                if param.name in np_param_hdrs_hash:
                    self.param_extract_indexes[np_param_hdrs_hash[param.name]] = pindex
                elif param.map in np_param_hdrs_hash:
                    self.param_extract_indexes[np_param_hdrs_hash[param.map]] = pindex
        #pdb.set_trace()
        self.np_param_hdrs_hash = np_param_hdrs_hash
    
    # Get valid preloaded data.
    def GetValidPreloadedData(self):
        trans_data_flag = np.full(shape=trans_data.shape, fill_value=False)
        for (param, param_values, pindex, permutation_index) in self.pruned_param_list:
            for pvalue in param_values:
                trans_data_flag[pindex] = trans_data_flag[pindex] | (
                    trans_data[pindex] == pvalue
                )
        for (param, param_values, pindex, permutation_index) in self.pruned_param_list:
            if pindex not in self.pruned_list_indexes:
                trans_data_flag[pindex] = np.full(
                    shape=trans_data_flag[pindex].shape, fill_value=True
                )
        trans_data_flag = trans_data_flag.transpose()
        valid_indexes = []
        for index, tdata in enumerate(trans_data_flag):
            if tdata.all():
                valid_indexes.append(index)
        # TODO

    def GetParamValKey(self, param_val):
        param_hash_key = tuple(param_val[self.param_extract_indexes].tolist())
        return param_hash_key

    def PushEvaluatedData(self, param_val, data):
        param_hash_key = self.GetParamValKey(param_val)
        self.param_data_hash[param_hash_key] = data

    def GetParamterCost(self, param_val):
        param_hash_key = self.GetParamValKey(param_val)
        if len(param_hash_key) > 0 and param_hash_key in self.param_data_hash:
            return self.param_data_hash[param_hash_key]
        return None

    # Get parameters full list which includes the parameters 
    # used only for ML model and unused parameters
    def GetParamsFullList(self, np_params):
        return np.append(np_params, self.unused_params_values)[
            self.rev_param_extract_indexes,
        ]

    # Get pre-evaluated parameters
    def GetPreEvaluatedParameters(self, samples, param_list):
        out_params = []
        param_data_sample_indexes = self.param_data_sample_indexes
        (common_samples, indexes, sel_indexes) = np.intersect1d(samples, param_data_sample_indexes, return_indices=True)
        if len(sel_indexes) == 0:
            return ([], np.array([]))
        selected = self.param_data[sel_indexes]
        rows = selected.shape[0]
        unused_cols = len(self.unused_params_values)
        rev_param = np.repeat(self.unused_params_values, rows, axis=0).reshape(unused_cols, rows).transpose()
        all_params = np.concatenate((selected, rev_param), axis=1)
        out_params = all_params.transpose()[self.rev_param_extract_indexes].transpose()
        return (indexes, out_params)


# Get object of evaluate
def GetObject(framework):
    obj = ParamData(framework)
    return obj

