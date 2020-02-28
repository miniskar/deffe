import os
import pdb
import numpy as np

class Parameters:
    def __init__(self, config):
        self.config = config
        self.type_knob = 1
        self.type_scenario = 2

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
        return (self.selected_params, self.selected_pruned_params, total_permutations)

    def GetPrunedSelectedValues(self, parameters, pruned_param_list):
        indexes = []
        for (param, param_values, pindex) in pruned_param_list:
            indexes.append(pindex)
        return [param[indexes,] for param in parameters]

    def GetPermutationSelection(self, nd_index):
        index = len(self.selected_params)-1
        out_dim_list = []
        for (param, param_values, pindex) in reversed(self.selected_params):
            dim_index = int(nd_index / self.indexing[index])
            out_dim_list.append(dim_index)
            nd_index = nd_index % self.indexing[index]
            index = index - 1
        out_dim_list.reverse()
        return out_dim_list

    # Make sure that all values in the numpy 2d array are values
    def GetNormalizedParameters(self, nparams, selected_params=None):
        if selected_params == None:
            selected_params = self.selected_params
        min_list = []
        max_list = []
        for (param, param_values, pindex) in selected_params:
            min_list.append(np.min(np.array(param_values).astype('float'))) 
            max_list.append(np.max(np.array(param_values).astype('float')))
        min_list = np.array(min_list).astype('float')
        max_list = np.array(max_list).astype('float')
        nparams_out = nparams.astype('float')
        nparams_out = (nparams_out - min_list) / (max_list - min_list)
        if (nparams_out< 0.0).any():
            print("Error: Some data in the sample normalization is negative. Please define the ranges properly")
            print([index for index,param in enumerate(nparams_out) if (param<0.0).any()])
            pdb.set_trace()
            None
        if (nparams_out> 1.0).any():
            print("Error: Some data in the sample normalization is > 1.0. Please define the ranges properly")
            print([index for index, param in enumerate(nparams_out) if (param>1.0).any()])
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

    def GetParameters(self, samples, selected_params=None, with_indexing=False, with_normalize=False):
        if selected_params == None:
            selected_params = self.selected_params
        nparams = self.GetNumpyParameters(samples, selected_params, with_indexing=with_indexing)
        if with_normalize:
            nparams = self.GetNormalizedParameters(nparams, selected_params)
        return nparams

    def GetPrunedSelectedParams(self, param_list):
        return [(param, pvalues, pindex) for (param, pvalues, pindex) in param_list if len(pvalues) > 1]

    def GetHeaders(self, param_list):
        return [param.name for (param, pvalues, pindex) in param_list]
