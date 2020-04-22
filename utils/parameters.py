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
import os
import pdb
import numpy as np
import re


def IsNumber(x):
    allowed_types = [
        float,
        int,
        np.float64,
        np.float32,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
    ]
    if type(x) in allowed_types:
        return True
    return False


def IsFloat(x):
    float_pattern = (
        "^[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?$"
    )
    float_re = re.compile(float_pattern, re.VERBOSE)
    if IsNumber(x):
        return True
    if float_re.findall(x):
        return True
    else:
        return False


class Parameters:
    def __init__(self, config, framework):
        self.config = config
        self.type_knob = 1
        self.type_scenario = 2
        self.framework = framework
        self.min_list_params = {}
        self.max_list_params = {}

    # Entry method to get all permutations
    def Initialize(self, explore_groups):
        knobs = self.config.GetKnobs()
        scenarios = self.config.GetScenarios()
        param_groups = {}
        for knob in knobs:
            groups = knob.groups
            for grp in groups:
                if grp not in explore_groups:
                    continue
                if grp not in param_groups:
                    param_groups[grp] = {}
                knob_name_values = param_groups[grp]
                map_name = knob.name
                if map_name not in knob_name_values:
                    # Mapped knobs and common values
                    knob_name_values[map_name] = ([], [])
                knob_name_values[map_name][0].append((knob, self.type_knob))
                common_data = []
                common_data.extend(knob.values.values)
                common_data.extend(knob_name_values[map_name][1])
                del knob_name_values[map_name][1][:]
                knob_name_values[map_name][1].extend(common_data)
        for scn in scenarios:
            groups = scn.groups
            for grp in groups:
                if grp not in explore_groups:
                    continue
                if grp not in param_groups:
                    param_groups[grp] = {}
                scn_name_values = param_groups[grp]
                map_name = scn.name
                if map_name not in scn_name_values:
                    # Mapped scns and common values
                    scn_name_values[map_name] = ([], [])
                scn_name_values[map_name][0].append((scn, self.type_scenario))
                common_data = []
                common_data.extend(scn.values.values)
                common_data.extend(scn_name_values[map_name][1])
                del scn_name_values[map_name][1][:]
                scn_name_values[map_name][1].extend(common_data)
        output_params = []
        total_permutations = 1
        for grp in param_groups.keys():
            for map_name in param_groups[grp].keys():
                (param_list, param_values) = param_groups[grp][map_name]
                for (param, ptype) in param_list:
                    output_params.append((param, param_values, len(output_params)))
                    total_permutations = total_permutations * len(param_values)
        self.selected_params = output_params
        self.total_permutations = total_permutations
        self.indexing = []
        prev_dim_elements = 1
        self.param_all_values = []
        for (param, param_values, pindex) in self.selected_params:
            self.indexing.append(prev_dim_elements)
            prev_dim_elements = len(param_values) * prev_dim_elements
            self.param_all_values.append(len(param_values))
        self.prev_dim_elements = prev_dim_elements
        self.selected_pruned_params = self.GetPrunedSelectedParams(self.selected_params)
        for (param, param_values, pindex) in self.selected_params:
            is_numbers = self.IsParameterNumber(param_values)
            if is_numbers:
                minp = np.min(np.array(param_values).astype("float"))
                maxp = np.max(np.array(param_values).astype("float"))
                # print("1MinP: "+str(minp)+" maxP:"+str(maxp)+" name:"+param.map)
                self.UpdateMinMaxRange(param, minp, maxp)
        # print("Initialize Options:"+str(self.GetMinMaxToJSonData()))
        return (self.selected_params, self.selected_pruned_params, total_permutations)

    def IsParameterNumber(self, param_values):
        is_numbers = True
        for val in param_values:
            is_numbers = is_numbers & IsFloat(val)
        return is_numbers

    def GetPrunedSelectedValues(self, parameters, pruned_param_list):
        indexes = []
        for (param, param_values, pindex) in pruned_param_list:
            indexes.append(pindex)
        return [param[indexes,] for param in parameters]

    def GetPermutationSelection(self, nd_index):
        index = len(self.selected_params) - 1
        out_dim_list = []
        for (param, param_values, pindex) in reversed(self.selected_params):
            dim_index = int(nd_index / self.indexing[index])
            out_dim_list.append(dim_index)
            nd_index = nd_index % self.indexing[index]
            index = index - 1
        out_dim_list.reverse()
        return out_dim_list

    def UpdateMinMaxRange(self, param, minp, maxp):
        # print("Options update:"+param.map)
        # print("Before Options:"+str(self.GetMinMaxToJSonData()))
        if param.map in self.min_list_params:
            minp = min(minp, self.min_list_params[param.map])
        if param.map in self.max_list_params:
            maxp = max(maxp, self.max_list_params[param.map])
        self.min_list_params[param.map] = minp
        self.max_list_params[param.map] = maxp
        # print("Before Options:"+str(self.GetMinMaxToJSonData()))

    def GetMinMaxToJSonData(self):
        data = {}
        for (param, param_values, pindex) in self.selected_params:
            if param.map in data:
                continue
            enable = False
            if param.map in self.min_list_params:
                enable = True
            if param.map in self.min_list_params:
                enable = True
            if enable:
                data[param.map] = {}
                if param.map in self.min_list_params:
                    data[param.map]["min"] = self.min_list_params[param.map]
                if param.map in self.max_list_params:
                    data[param.map]["max"] = self.max_list_params[param.map]
        return data

    # Make sure that all values in the numpy 2d array are values
    def GetNormalizedParameters(self, nparams, selected_params=None):
        if selected_params == None:
            selected_params = self.selected_params
        min_list = []
        max_list = []
        for (param, param_values, pindex) in selected_params:
            min_list.append(self.min_list_params[param.map])
            max_list.append(self.max_list_params[param.map])
        min_list = np.array(min_list).astype("float")
        max_list = np.array(max_list).astype("float")
        nparams_out = nparams.astype("float")
        nparams_out = (nparams_out - min_list) / (max_list - min_list)
        if not self.framework.args.bounds_no_check:
            if (nparams_out < 0.0).any():
                print(
                    "Error: Some data in the sample normalization is negative. Please define the ranges properly"
                )
                print(
                    [
                        index
                        for index, param in enumerate(nparams_out)
                        if (param < 0.0).any()
                    ]
                )
                pdb.set_trace()
                None
            if (nparams_out > 1.0).any():
                print(
                    "Error: Some data in the sample normalization is > 1.0. Please define the ranges properly"
                )
                print(
                    [
                        index
                        for index, param in enumerate(nparams_out)
                        if (param > 1.0).any()
                    ]
                )
                pdb.set_trace()
                None
        return nparams_out

    def GetNumpyParameters(self, samples, selected_params=None, with_indexing=False):
        if selected_params == None:
            selected_params = self.selected_params
        nparams = np.empty(shape=[0, len(selected_params)])
        indexes = samples[0].tolist() + samples[1].tolist()
        for nd_index in indexes:
            out_dim_list = self.GetPermutationSelection(nd_index)
            param_sel_list = []
            for (param, param_values, index) in selected_params:
                param_value_index = out_dim_list[index]
                value = param_values[param_value_index]
                if with_indexing and type(value) == str:
                    value = param_value_index
                param_sel_list.append(value)
            nparams = np.concatenate((nparams, [np.array(param_sel_list)]))
        return nparams

    def GetParameters(
        self, samples, selected_params=None, with_indexing=False, with_normalize=False
    ):
        if selected_params == None:
            selected_params = self.selected_params
        nparams = self.GetNumpyParameters(
            samples, selected_params, with_indexing=with_indexing
        )
        if with_normalize:
            nparams = self.GetNormalizedParameters(nparams, selected_params)
        return nparams

    def GetPrunedSelectedParams(self, param_list):
        return [
            (param, pvalues, pindex)
            for (param, pvalues, pindex) in param_list
            if len(pvalues) > 1
        ]

    def GetHeaders(self, param_list):
        return [param.name for (param, pvalues, pindex) in param_list]

    def CreateRunScript(self, script, run_dir, param_pattern, param_dict):
        with open(script, "r") as rfh, open(
            os.path.join(run_dir, os.path.basename(script)), "w"
        ) as wfh:
            lines = rfh.readlines()
            for line in lines:
                wline = param_pattern.sub(
                    lambda m: param_dict[re.escape(m.group(0))], line
                )
                wfh.write(wline)
            rfh.close()
            wfh.close()

    def GetParamHash(self, param_val, param_list=None):
        if param_list == None:
            param_list = self.selected_params
        param_hash = {}
        index = 0
        for (param, param_values, pindex) in param_list:
            param_key = "${" + param.name + "}"
            if index >= len(param_val):
                pdb.set_trace()
                None
            param_hash[param_key] = param_val[index]
            param_key = "${" + param.map + "}"
            if param.name != param.map:
                if param_key in param_hash:
                    print(
                        "[Error] Multiple map_name(s):"
                        + param.map
                        + " used in the evaluation"
                    )
                param_hash[param_key] = param_val[index]
            index = index + 1
        param_dict = dict((re.escape(k), v) for k, v in param_hash.items())
        param_pattern = re.compile("|".join(param_dict.keys()))
        return (param_pattern, param_hash, param_dict)