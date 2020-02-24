import os
import pdb

class ParamsCost:
    def __init__(self, config):
        self.config = config
        self.type_wl_param = 0
        self.type_wl_knob = 1
        self.type_sys_knob = 2
        self.InitDataStructures()

    def GetUniqueName(self, sys_wl, knob_param):
        return sys_wl.name+"::"+knob_param.name

    def GetEvaluateSpecificParameters(self, wl_name=None, sys_name=None):
        all_options = {}
        for param_name, pobj in self.unique_param_hash.items():
            (system_wl, knob_param, type) = pobj
            if (type == self.type_wl_param or
               type == self.type_wl_knob) and wl_name != None and
               system_wl.name != wl_name:
                continue
            if type == self.type_sys_knob and sys_name != None and
               system_wl.name != sys_name:
                continue
            print("Values length: "+str(len(pobj.values.values)))
            values = list(set(pobj.values.values))
            all_options[param_name] = (values, system_wl, knob_param, type)
            print(values)
            print("Len:"+str(len(values)))
        return all_options

    # Get permutation for all options
    def GetPermutations(self, sel_options):
        def func(sel_options, group=[], result=[]):
            if not sel_options:
                result.append(group)
                return
            first, rest = sel_options[0], sel_options[1:]
            for letter in first:
                func(rest, group + [letter], result)
        result = []
        func(sel_options, result=result)
        return result

    def GetFixedVariableOptions(self, fix_opts, all_options, opt_val={}):
        var_keys = []
        var_sel_options = []
        for k, v in all_options.items():
            if type(v) == list:
                var_keys.append(k)
                var_sel_options.append(v)
            else:
                if v == "":
                    fix_opts.append(k)
                else:
                    fix_opts.append(k+"="+v)
                    opt_val[k] = v
        return (fix_opts, var_keys, var_sel_options)

    # Entry method to get all permutations
    def InitDataStructures(self):
        systems = self.config.GetSystems()
        workloads = self.config.GetWorkloads()
        costs = self.config.GetCosts()
        system_knobs = {}
        wl_params = {}
        wl_knobs = {}
        param_hash = {} #key : [[#1] [#2]] #1: details of param #2: unique values set
        unique_param_hash = {}
        for system in systems:
            for knob in system.knobs:
                if system.name not in system_knobs:
                    system_knobs[system.name] = []
                system_knobs[system.name].append(knob)
                if knob.name not in param_hash:
                    param_hash[knob.name] = [[], []]
                param_hash[knob.name][0].append((system, knob, self.type_sys_knob))
                unique_param_hash[self.GetUniqueName(system, knob)] = (system, knob, self.type_sys_knob)
        for wl in workloads:
            for knob in wl.knobs:
                if wl.name not in wl_knobs:
                    wl_knobs[wl.name] = []
                wl_knobs[wl.name].append(knob)
                if knob.name not in param_hash:
                    param_hash[knob.name] = [[], []]
                param_hash[knob.name][0].append((wl, knob, self.type_wl_knob))
                unique_param_hash[self.GetUniqueName(wl, knob)] = (wl, knob, self.type_wl_knob)
        for wl in workloads:
            for param in wl.parameters:
                if wl.name not in wl_params:
                    wl_params[wl.name] = []
                wl_params[wl.name].append(param)
                if param.name not in param_hash:
                    param_hash[param.name] = [[], []]
                param_hash[param.name][0].append((wl, param, self.type_wl_param))
                unique_param_hash[self.GetUniqueName(wl, param)] = (wl, param, self.type_wl_param)
        self.system_knobs = system_knobs
        self.wl_params = wl_params
        self.wl_knobs = wl_knobs
        self.param_hash = param_hash
        self.unique_param_hash = unique_param_hash
        all_options = {}
        for param_name, v in param_hash.items():
            common_data = []
            for data in v[0]:
                #(system/wl, knob/param, type) = data
                pobj = data[1]
                print("Values length: "+str(len(pobj.values.values)))
                common_data.extend(pobj.values.values)
            v[1].extend(list(set(common_data)))
            all_options[param_name] = v[1]
            print(v[1])
            print("Len:"+str(len(v[1])))
        self.all_options = all_options


    def GetAllParameterPermutations(self):
        print("GetFixvars")
        opt_val_hash = {}
        fix_opts = []
        (fix_opts, var_keys, var_sel_options) = self.GetFixedVariableOptions(fix_opts, self.all_options, opt_val_hash)
        print("GetPermutations")
        all_output = self.GetPermutations(var_sel_options)
        print("Set outputs")
        self.all_output = all_output

