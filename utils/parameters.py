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
        if len(explore_groups) == 0:
            explore_groups['all'] = 1
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
                    output_params.append((param, param_values))
                    total_permutations = total_permutations * len(param_values)
        self.selected_params = output_params
        self.total_permutations = total_permutations
        self.indexing = []
        prev_dim_elements = 1
        for (param, param_values) in self.selected_params:
            self.indexing.append(prev_dim_elements)
            prev_dim_elements = len(param_values) * prev_dim_elements
        self.prev_dim_elements = prev_dim_elements
        return (output_params, total_permutations)

    def GetPermutationSelection(self, nd_index):
        index = len(self.selected_params)-1
        out_dim_list = []
        for (param, param_values) in reversed(self.selected_params):
            dim_index = int(nd_index / self.indexing[index])
            out_dim_list.append(dim_index)
            nd_index = nd_index % self.indexing[index]
            index = index - 1
        out_dim_list.reverse()
        return out_dim_list


    def GetParameters(self, samples):
        indexes = samples[0].tolist() + samples[1].tolist()
        nparams = np.empty(shape=[0, len(self.selected_params)])
        min_list = []
        max_list = []
        for (param, param_values) in self.selected_params:
            min_list.append(np.min(np.array(param_values).astype('float'))) 
            max_list.append(np.max(np.array(param_values).astype('float')))
        min_list = np.array(min_list).astype('float')
        max_list = np.array(max_list).astype('float')
        for nd_index in indexes:
            out_dim_list = self.GetPermutationSelection(nd_index)
            param_sel_list = []
            index = 0
            for (param, param_values) in self.selected_params:
                value = param_values[out_dim_list[index]]
                param_sel_list.append(value)
                index = index + 1
            nparams = np.concatenate((nparams, [np.array(param_sel_list)]))
        nparams = nparams.astype('float')
        nparams = (nparams - min_list) / (max_list - min_list)
        return nparams
