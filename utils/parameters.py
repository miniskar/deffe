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
import sys
from read_config import *

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
    def __init__(self, config=None, framework=None):
        self.config = config
        self.type_knob = 1
        self.type_scenario = 2
        self.framework = framework
        self.min_list_params = {}
        self.max_list_params = {}
        self.is_values_string = {}

    # Entry method to get all permutations
    def Initialize(self, explore_groups):
        def get_explore_groups_hash():
            exp_hash = {}
            for grp in explore_groups:
                grp_fields = re.split(r'\s*::\s*', grp)
                grp_key = grp_fields[0]
                if grp_key not in exp_hash:
                    exp_hash[grp_key] = []
                if len(grp_fields) > 1:
                    # Use second level delimiter Semicolon (;)
                    values = DeffeConfigValues(grp_fields[1], ';').values
                    exp_hash[grp_key].extend(values)
            return exp_hash
        exp_grp_hash = get_explore_groups_hash()
        knobs = self.config.GetKnobs()
        scenarios = self.config.GetScenarios()
        knob_scns_list = [ (k, self.type_knob) for k in knobs ]
        knob_scns_list.extend([ (s, self.type_scenario) for s in scenarios ])
        param_groups = {}
        for (knob_scn, ks_type) in knob_scns_list:
            groups = knob_scn.groups
            for grp in groups:
                if grp not in exp_grp_hash:
                    # Check if it has only "all", but no other group is configured
                    if knob_scn.groups_configured: 
                        continue
                if grp not in param_groups:
                    param_groups[grp] = {}
                ks_name_values = param_groups[grp]
                ks_name = knob_scn.name
                if ks_name not in ks_name_values:
                    # Mapped knobs and common values
                    ks_name_values[ks_name] = ([], [])
                ks_name_values[ks_name][0].append((knob_scn, ks_type))
                common_data = []
                if grp in exp_grp_hash and len(exp_grp_hash[grp]) > 0:
                    common_data.extend(exp_grp_hash[grp])
                else:
                    common_data.extend(knob_scn.values.values)
                    common_data.extend(ks_name_values[ks_name][1])
                    del ks_name_values[ks_name][1][:]
                ks_name_values[ks_name][1].extend(common_data)
        param_unique_hash = {}
        for grp in param_groups.keys():
            for ks_name in param_groups[grp].keys():
                print("Grp:{} ks:{}".format(grp, ks_name))
                (param_list, param_values) = param_groups[grp][ks_name]
                for (param, ptype) in param_list:
                    if param.name in param_unique_hash:
                        # Already existing 
                        if ks_name == grp:
                            # Give priority to this
                            del param_unique_hash[param.name]
                            param_unique_hash[param.name] = (param, param_values)
                    else:
                        param_unique_hash[param.name] = (param, param_values)
        total_permutations = 1
        output_params = []
        for param_name, (param, param_values) in \
                sorted(param_unique_hash.items(), key=lambda x: x[0]):
            if param.base == None:
                param.base = 0
            else:
                param.base = param_values.index(param.base)
            output_params.append((param, param_values, 
                        len(output_params), total_permutations))
            total_permutations = len(param_values) * total_permutations 
        self.selected_params = output_params
        self.total_permutations = total_permutations
        self.selected_pruned_params = \
                 self.GetPrunedSelectedParams(self.selected_params)
        for (param, param_values, pindex, permutation_index) in \
                self.selected_params:
            is_numbers = self.IsParameterNumber(param_values)
            if is_numbers:
                minp = np.min(np.array(param_values).astype("float"))
                maxp = np.max(np.array(param_values).astype("float"))
                # print("1MinP: "+str(minp)+" maxP:"+str(maxp)+" name:"+param.map)
                self.UpdateMinMaxRange(param, minp, maxp)
                self.is_values_string[param.map] = False
            else:
                minp = 0
                maxp = len(param_values)
                self.UpdateMinMaxRange(param, minp, maxp)
                self.is_values_string[param.map] = True
        # print("Initialize Options:"+str(self.GetMinMaxToJSonData()))
        print("******* All considered parameters *******")
        for (param, param_values, pindex, permutation_index) in self.selected_params:
            print("{}: {} = {}".format(pindex, param.name, param_values))
        print("******* Variable parameters *******")
        for (param, param_values, pindex, permutation_index) in self.selected_pruned_params:
            print("{}: {} = {}".format(pindex, param.name, param_values))
        #print(self.selected_pruned_params)
        print("Total permutations:"+str(total_permutations))
        return (self.selected_params, self.selected_pruned_params, total_permutations)

    def IsParameterNumber(self, param_values):
        is_numbers = True
        for val in param_values:
            is_numbers = is_numbers & IsFloat(val)
        return is_numbers

    def GetPrunedSelectedValues(self, parameters, pruned_param_list):
        indexes = []
        for (param, param_values, pindex, permutation_index) in pruned_param_list:
            indexes.append(pindex)
        return [param[indexes,] for param in parameters]

    def EncodePermutationFromArray(self, sel_param_indices, selected_params=None):
        if selected_params == None:
            selected_params = self.selected_pruned_params
        perm_index = 0
        for index, (param, param_values, pindex, permutation_index) in enumerate(selected_params):
            val_index = sel_param_indices[index] 
            perm_index = perm_index + val_index * permutation_index
        return perm_index

    def EncodePermutation(self, sel_param_hash, selected_params=None):
        if selected_params == None:
            selected_params = self.selected_params
        perm_index = 0
        for (param, param_values, pindex, permutation_index) in reversed(selected_params):
            val_index = 0 
            if param.name in sel_param_hash:
                val = sel_param_hash[param.name]
                val_index = param_values.index(val)
            perm_index = perm_index + val_index * permutation_index
        return perm_index

    def GetPermutationKeyValues(self, nd_index):
        val = self.GetPermutationSelection(nd_index)
        val_hash = { param.name:param_values[val[index]] 
            for index, (param, param_values, pindex, permutation_index) in enumerate(self.selected_params) }
        return val_hash

    def GetPermutationSelection(self, nd_index, selected_params=None):
        if selected_params == None:
            selected_params = self.selected_params
        out_dim_list = []
        for (param, param_values, pindex, permutation_index) in reversed(selected_params):
            dim_index = int(nd_index / permutation_index)
            out_dim_list.append(dim_index)
            nd_index = nd_index % permutation_index
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
        for (param, param_values, pindex, permutation_index) in self.selected_params:
            if param.map in data:
                continue
            enable = False
            if param.map in self.min_list_params:
                enable = True
            if param.map in self.min_list_params:
                enable = True
            if enable:
                data[param.map] = {}
                data[param.map]["count"] = len(param_values)
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
        nparams_t = nparams.transpose()
        for index, (param, param_values, pindex, permutation_index) in enumerate(selected_params):
            if param.map not in self.min_list_params:
                print("[Error] key:{} not found in min_list_params".format(param.map))
                pdb.set_trace()
                None
            min_list.append(self.min_list_params[param.map])
            max_list.append(self.max_list_params[param.map])
            if self.is_values_string[param.map]:
                val_hash = { k:cindex for cindex, k in enumerate(param_values) }
                for cindex in range(len(nparams_t[index])):
                    nparams_t[index][cindex] = val_hash[nparams_t[index][cindex]]
        min_list = np.array(min_list).astype("float")
        max_list = np.array(max_list).astype("float")
        nparams_t = nparams_t.transpose()
        nparams_out = nparams_t.astype("float")
        nparams_out = (nparams_out - min_list) / (max_list - min_list)
        if self.framework != None and \
            not self.framework.args.bounds_no_check:
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
        indexes = samples
        for nd_index in indexes:
            out_dim_list = self.GetPermutationSelection(nd_index)
            param_sel_list = []
            for (param, param_values, index, permutation_index) in selected_params:
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
            (param, pvalues, pindex, permutation_index)
            for (param, pvalues, pindex, permutation_index) in param_list
            if len(pvalues) > 1
        ]

    def GetHeaders(self, param_list):
        return [param.name for (param, pvalues, pindex, permutation_index) in param_list]

    def CreateRunScript(self, script, run_dir, param_pattern, param_val_with_escapechar_hash):
        if not os.path.exists(script):
            with open(script, "w") as wfh:
                wfh.write("")
                wfh.close()
        with open(script, "r") as rfh, open(
            os.path.join(run_dir, os.path.basename(script)), "w"
        ) as wfh:
            lines = rfh.readlines()
            for line in lines:
                wline = param_pattern.sub(
                    lambda m: param_val_with_escapechar_hash.get(re.escape(m.group(0)), m.group(0)), line
                )
                wfh.write(wline)
            rfh.close()
            wfh.close()

    def GetParamHash(self, param_val, param_list=None):
        if param_list == None:
            param_list = self.selected_params
        param_val_hash = {}
        index = 0
        for (param, param_values, pindex, permutation_index) in param_list:
            param_key1 = "${" + param.name + "}"
            param_key2 = "$" + param.name+""
            if index >= len(param_val):
                pdb.set_trace()
                None
            param_val_hash[param_key1] = param_val[index]
            param_val_hash[param_key2] = param_val[index]
            param_key1 = "${" + param.map + "}"
            param_key2 = "$" + param.map+""
            if param.name != param.map:
                if param_key1 in param_val_hash:
                    print(
                        "[Error] Multiple map_name(s):"
                        + param.map
                        + " used in the evaluation"
                    )
                param_val_hash[param_key1] = param_val[index]
                if param_key2 in param_val_hash:
                    print(
                        "[Error] Multiple map_name(s):"
                        + param.map
                        + " used in the evaluation"
                    )
                param_val_hash[param_key2] = param_val[index]
            index = index + 1
        param_val_with_escapechar_hash = dict((re.escape(k), v) 
                for k, v in param_val_hash.items())
        param_pattern = re.compile(r'\$[{]?\b[a-zA-Z0-9_]+\b[}]?')
        return (param_pattern, param_val_hash, param_val_with_escapechar_hash)


if __name__ == "__main__":
    framework_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    framework_env = os.getenv("DEFFE_DIR")
    if framework_env == None:
        os.environ["DEFFE_DIR"] = framework_path
    sys.path.insert(0, os.getenv("DEFFE_DIR"))
    sys.path.insert(0, os.path.join(framework_path, "framework"))
    sys.path.insert(0, os.path.join(framework_path, "utils"))
    from read_config import *
    config = DeffeConfig("config_small_sampling.json")
    explore_groups = config.GetExploration().exploration_list[0].groups
    params = Parameters(config, None)
    params.Initialize(explore_groups)
    val = params.GetPermutationSelection(1234)
    val_hash = { param.name:param_values[val[index]] 
        for index, (param, param_values, pindex, permutation_index) in enumerate(params.selected_params) }
    print(val_hash)
    index = params.EncodePermutationHash(val_hash)
    if index != 1234:
        print("Error")
    print(index)
