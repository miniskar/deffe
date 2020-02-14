import os
import pdb

class ParamsCost:
    def __init__(self, config):
        self.config = config
        self.type_wl_param = 0
        self.type_wl_knob = 1
        self.type_sys_knob = 2
        self.CombinedParamsCosts()

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

    def CombinedParamsCosts(self):
        systems = self.config.GetSystems()
        workloads = self.config.GetWorkloads()
        costs = self.config.GetCosts()
        system_knobs = {}
        wl_params = {}
        wl_knobs = {}
        param_hash = {}
        for system in systems:
            for knob in system.knobs:
                if system.name not in system_knobs:
                    system_knobs[system.name] = []
                system_knobs[system.name].append(knob)
                if knob.name not in param_hash:
                    param_hash[knob.name] = [[], []]
                param_hash[knob.name][0].append((system, knob, self.type_sys_knob))
        for wl in workloads:
            for knob in wl.knobs:
                if wl.name not in wl_knobs:
                    wl_knobs[wl.name] = []
                wl_knobs[wl.name].append(knob)
                if knob.name not in param_hash:
                    param_hash[knob.name] = [[], []]
                param_hash[knob.name][0].append((wl, knob, self.type_wl_knob))
        for wl in workloads:
            for param in wl.parameters:
                if wl.name not in wl_params:
                    wl_params[wl.name] = []
                wl_params[wl.name].append(param)
                if param.name not in param_hash:
                    param_hash[param.name] = [[], []]
                param_hash[param.name][0].append((wl, param, self.type_wl_param))
        self.system_knobs = system_knobs
        self.wl_params = wl_params
        self.wl_knobs = wl_knobs
        self.param_hash = param_hash
        opt_val_hash = {}
        all_options = {}
        fix_opts = []
        for param_name, v in param_hash.items():
            common_data = []
            for data in v[0]:
                pobj = data[1]
                print("Values length: "+str(len(pobj.values.values)))
                common_data.extend(pobj.values.values)
            v[1].extend(list(set(common_data)))
            all_options[param_name] = v[1]
            print(v[1])
            print("Len:"+str(len(v[1])))
        print("GetFixvars")
        (fix_opts, var_keys, var_sel_options) = self.GetFixedVariableOptions(fix_opts, all_options, opt_val_hash)
        print("GetPermutations")
        all_output = self.GetPermutations(var_sel_options)
        print("Set outputs")
        self.all_output = all_output

