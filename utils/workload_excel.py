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
import xlsxwriter
import numpy as np
import re
import pdb
import sys
import time
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from multicore_run import *
# Is string floating point number
float_pattern = "^[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?$"
int_pattern = "^([0-9]*)$"
int_re = re.compile(int_pattern)
float_re = re.compile(float_pattern, re.VERBOSE)
re_comma_comp = re.compile(r",")


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
    if IsNumber(x):
        return True
    if float_re.findall(x):
        return True
    else:
        return False


class Workload:
    def __init__(self):
        self.param_exclude_re = re.compile(
            r"eval_id|^options_rows|^time$|^pid$|interface|index|machine|^cpu$|^CounterCycles$|WorkDir|area|power",
            re.IGNORECASE,
        )
        self.cost_re = re.compile(r"area|power|cycle|obj_fn_", re.IGNORECASE)
        self.cost_re = re.compile(r"leakage power|cpu_cycles|obj_fn_", re.IGNORECASE)
        self.headers_data = []
        self.data_lines = []
        self.headers_index_hash = {}
        self.param_indexes = []
        self.cost_indexes = []
        self.data_hash = {}
        self.csv_filename = None

    def GetHeaders(self, specific_columns=None):
        if specific_columns == None:
            specific_columns = range(0, len(self.headers_data))
        return [self.headers_data[index] for index in specific_columns]

    def GetParamCostHeaders(self):
        return self.GetHeaders(self.param_indexes + self.cost_indexes)

    def GetParamIndexes(self, hdrs):
        return [
            index
            for index, hdr in enumerate(hdrs)
            if not self.param_exclude_re.search(hdr) and not self.cost_re.search(hdr)
        ]

    def IsHdrExist(self, name):
        if name in self.headers_index_hash:
            return True
        return False

    def GetCostIndexes(self, hdrs):
        return [index for index, hdr in enumerate(hdrs) if self.cost_re.search(hdr)]

    def GetHdrIndex(self, index_name):
        return self.headers_index_hash[index_name]

    # Remove whitespaces in line
    def RemoveWhiteSpaces(self, line):
        line = re.sub(r"\r", "", line)
        line = re.sub(r"\n", "", line)
        line = re.sub(r"^\s*$", "", line)
        line = re.sub(r"^\s*", "", line)
        line = re.sub(r"\s*$", "", line)
        line = re.sub(r"kdd_cup-", "", line)
        return line

    def GetLogDataAtIndexes(self, data, indexes):
        return [str(np.log(float(data[i]))) for i in indexes]

    def GetDataAtIndexesWithCheck(self, data, indexes):
        return [data[i] if i != -1 else "0" for i in indexes]

    def GetDataAtIndexes(self, data, indexes):
        return [data[i] for i in indexes]
        # return map(lambda i: data[i], indexes)

    def GetValueNormAtIndexes(self, data, index, indexes, exclusions={}):
        # For non-exclusion indexes, get its continuous domain unique value.
        # For exclusion indexes, get the data as it is
        unique_mean_values_hash = self.unique_mean_values_hash
        return [
            unique_mean_values_hash[i][data[i]] if i not in exclusions else data[i]
            for i in indexes
        ]
        # return map(lambda i: unique_values_hash[i][data[i]], indexes)

    def GetValueIndexAtIndexes(self, data, indexes, exclusions={}):
        # For non-exclusion indexes, get its continuous domain unique value.
        # For exclusion indexes, get the data as it is
        unique_values_hash = self.unique_values_hash
        return [
            unique_values_hash[i][data[i]] if i not in exclusions else data[i]
            for i in indexes
        ]
        # return map(lambda i: unique_values_hash[i][data[i]], indexes)

    def GetValidHdrs(self, excluded_hdrs):
        hdrs = self.headers_data
        valid_hdrs = []
        for hdr in hdrs:
            if hdr not in excluded_hdrs:
                valid_hdrs.append(hdr)
        return valid_hdrs

    def GroupsData(self, group_hdrs, data_lines=None):
        group_data = {}
        indexes = []
        for gname in group_hdrs:
            hindex = self.headers_index_hash[gname]
            indexes.append(hindex)
        global_key = "plot"
        if data_lines == None:
            data_lines = self.data_lines
        for rdata in data_lines:
            key = global_key
            if len(indexes) == 1:
                key = rdata[indexes[0]]
            elif len(indexes) != 0:
                key = tuple(self.GetDataAtIndexes(rdata, indexes))
            if key not in group_data:
                group_data[key] = []
            group_data[key].append(rdata)
        return group_data

    def SetColumn(self, colname, value):
        col_index = colname
        if len(self.headers_data) > 0:
            col_index = self.GetHdrIndex(colname)
        for rdata in self.data_lines:
            rdata[col_index] = value

    def SortData(self, data, dtype=float, field_index=0):
        if type(field_index) == str:
            field_index = self.GetHdrIndex(field_index)
        if dtype == int or dtype == "int":
            data = sorted(data, key=lambda tup: int(tup[field_index]))
        elif dtype == str or dtype == "str":
            data = sorted(data, key=lambda tup: str(tup[field_index]))
        elif IsFloat(data[0][field_index]):
            data = sorted(data, key=lambda tup: float(tup[field_index]))
        else:
            data = sorted(data, key=lambda tup: tup[field_index])
        return data

    def SortWorkload(self, data=None, dtype=float, field_index=0):
        if type(field_index) == str:
            field_index = self.GetHdrIndex(field_index)
        if data == None:
            data = self.GetData()
        data = self.SortData(data, dtype, field_index)
        workload = Workload()
        workload.InitializeWorkload(self.GetHeaders(), data)
        return workload

    def InitializeWorkload(self, hdrs, data):
        self.InitializeHeaders(hdrs)
        self.InitializeData(data)
        for rdata in data:
            self.HashParamCostData(rdata)

    def InitializeHeaders(self, hdrs):
        # Set param_indexes and cost_indexes
        self.headers_data.extend(hdrs)
        for index, hdr in enumerate(self.headers_data):
            self.headers_index_hash[hdr] = index
        self.param_indexes.extend(self.GetParamIndexes(self.headers_data))
        self.cost_indexes.extend(self.GetCostIndexes(self.headers_data))

    def InitializeData(self, data_lines):
        self.data_lines = data_lines
        self.unique_values = [
            list(set(self.GetSingleColumn(col)))
            for col, hdr in enumerate(self.headers_data)
        ]
        self.max_min_values = []
        for hdata in self.unique_values:
            if len(hdata) > 0 and IsNumber(hdata[0]):
                self.max_min_values.append((max(hdata), min(hdata)))
            elif len(hdata) > 0 and IsFloat(hdata[0]):
                fl_data = list(map(float, hdata))
                self.max_min_values.append((max(fl_data), min(fl_data)))
            else:
                self.max_min_values.append((0, 0))
        self.unique_mean_values_hash = [
            {
                val: self.GetNormValue(val, self.max_min_values[i])
                for index, val in enumerate(hdata)
            }
            for i, hdata in enumerate(self.unique_values)
        ]
        self.unique_values_hash = [
            {val: index for index, val in enumerate(hdata)}
            for hdata in self.unique_values
        ]

    def HashParamCostData(self, data_fields):
        tup_param = tuple(self.GetDataAtIndexes(data_fields, self.param_indexes))
        cost_data = self.GetDataAtIndexes(data_fields, self.cost_indexes)
        if tup_param not in self.data_hash:
            self.data_hash[tup_param] = cost_data
            return True
        else:
            return False

    def ReadPreEvaluatedWorkloads(self, filename, count=-1):
        print("Reading file: " + filename)

        def GetSplitFields(line):
            line = self.RemoveWhiteSpaces(line)
            if line == "":
                return []
            data_fields_core = None
            if re_comma_comp.search(line):
                data_fields_core = re.split(r"\s*,\s*", line)
            else:
                data_fields_core = re.split(r"\s+", line)
            return data_fields_core

        def ReadParametersAndCost(line):
            def FormatData(data):
                hd = {
                    "kb": 1024,
                    "mb": 1024 * 1024,
                    "gb": 1024 * 1024 * 1024,
                }
                suffix = data[-2:].lower()
                if suffix in hd:
                    return str(float(data[:-2]) * hd[suffix])
                return data

            data_fields_core = GetSplitFields(line)
            if data_fields_core == []:
                return []
            data_fields = [FormatData(data) for data in data_fields_core]
            if not self.HashParamCostData(data_fields):
                # print("Tuple exist:"+str(tup_param)+" prev_cost:"+str(self.data_hash[tup_param])+" prev_cost:"+str(cost_data))
                return []
            return data_fields

        with open(filename) as fh:
            all_lines = fh.readlines()
            if len(all_lines) == 0:
                return
            self.InitializeHeaders(GetSplitFields(all_lines[0]))
            if count != -1:
                all_lines = all_lines[0 : count + 1]
            print(
                "Params:"
                + str(
                    [
                        str(index) + ":" + self.headers_data[index]
                        for index in self.param_indexes
                    ]
                )
            )
            print(
                "Cost:"
                + str(
                    [
                        str(index) + ":" + self.headers_data[index]
                        for index in self.cost_indexes
                    ]
                )
            )
            # self.data_lines = ThreadPool(GetCPUCount()).map(ReadParametersAndCost, all_lines[1:])
            data_lines = [ReadParametersAndCost(line) for line in all_lines[1:]]
            data_lines = [data for data in data_lines if len(data) != 0]
            self.InitializeData(data_lines)
            print("Total records: " + str(len(self.data_lines)))

    def GetNormValue(self, val, max_min):
        if IsNumber(val):
            (max_val, min_val) = max_min
            if max_val == min_val:
                return val
            norm = (float(val) - float(min_val)) / (float(max_val) - float(min_val))
            return norm
        elif IsFloat(val):
            (max_val, min_val) = max_min
            if max_val == min_val:
                return val
            norm = (float(val) - float(min_val)) / (float(max_val) - float(min_val))
            return norm
        else:
            return val

    def ExcludeData(self, specific_columns, data_lines=None):
        if data_lines == None:
            data_lines = self.data_lines
        indexes = list(specific_columns.keys())
        if type(indexes[0]) == str or (
            sys.version_info[0] < 3 and type(indexes[0]) == unicode
        ):
            indexes = [
                self.headers_index_hash[col]
                for col in specific_columns.keys()
                if col in self.headers_index_hash
            ]

        rdata_list = []
        non_rdata_list = []
        for rdata in data_lines:
            found = True
            for index, col in enumerate(specific_columns.keys()):
                if rdata[indexes[index]] not in specific_columns[col]:
                    found = False
                    break
            if found:
                rdata_list.append(rdata)
            else:
                non_rdata_list.append(rdata)
        return rdata_list, non_rdata_list

    def GetSingleColumn(self, col, data=None):
        if data == None:
            data = self.data_lines
        return map(lambda x: x[col], data)

    def Get2DDataWithIndexing(self, specific_columns, data_lines=None, exclusions={}):
        if data_lines == None:
            data_lines = self.data_lines
        if type(specific_columns[0]) == str or (
            sys.version_info[0] < 3 and type(specific_columns[0]) == unicode
        ):
            indexes = [
                self.headers_index_hash[col]
                for col in specific_columns
                if col in self.headers_index_hash
            ]
            return [
                self.GetValueIndexAtIndexes(rdata, indexes, exclusions)
                for rdata in data_lines
            ]
        return [
            self.GetValueIndexAtIndexes(rdata, specific_columns, exclusions)
            for rdata in data_lines
        ]

    def Get2DDataWithNormalization(
        self, specific_columns, data_lines=None, exclusions={}
    ):
        if data_lines == None:
            data_lines = self.data_lines
        if type(specific_columns[0]) == str or (
            sys.version_info[0] < 3 and type(specific_columns[0]) == unicode
        ):
            indexes = [
                self.headers_index_hash[col]
                for col in specific_columns
                if col in self.headers_index_hash
            ]
            return [
                self.GetValueNormAtIndexes(rdata, index, indexes, exclusions)
                for index, rdata in enumerate(data_lines)
            ]
        return [
            self.GetValueNormAtIndexes(rdata, index, specific_columns, exclusions)
            for index, rdata in enumerate(data_lines)
        ]

    def Get2DLogData(self, specific_columns, data_lines=None):
        if data_lines == None:
            data_lines = self.data_lines
        if type(specific_columns[0]) == str or (
            sys.version_info[0] < 3 and type(specific_columns[0]) == unicode
        ):
            indexes = [
                self.headers_index_hash[col]
                for col in specific_columns
                if col in self.headers_index_hash
            ]
            return [self.GetLogDataAtIndexes(rdata, indexes) for rdata in data_lines]
        return [
            self.GetLogDataAtIndexes(rdata, specific_columns) for rdata in data_lines
        ]

    def Get2DData(self, specific_columns, data_lines=None):
        if data_lines == None:
            data_lines = self.data_lines
        if type(specific_columns[0]) == str or (
            sys.version_info[0] < 3 and type(specific_columns[0]) == unicode
        ):
            indexes = [
                self.headers_index_hash[col]
                for col in specific_columns
                if col in self.headers_index_hash
            ]
            return [self.GetDataAtIndexes(rdata, indexes) for rdata in data_lines]
        return [self.GetDataAtIndexes(rdata, specific_columns) for rdata in data_lines]

    def Get2DDataWithCheck(self, specific_columns, data_lines=None):
        if data_lines == None:
            data_lines = self.data_lines
        if type(specific_columns[0]) == str or (
            sys.version_info[0] < 3 and type(specific_columns[0]) == unicode
        ):
            indexes = [
                self.headers_index_hash[col]
                for col in specific_columns
                if col in self.headers_index_hash
            ]
            return [
                self.GetDataAtIndexesWithCheck(rdata, indexes) for rdata in data_lines
            ]
        return [
            self.GetDataAtIndexesWithCheck(rdata, specific_columns)
            for rdata in data_lines
        ]

    def GetData(self):
        return self.data_lines

    def Get2DParametersCostData(self):
        indexes = self.param_indexes + self.cost_indexes
        return self.Get2DData(indexes)

    def GetCostObjects(self, params):
        if tuple(params) in self.data_hash:
            return list(self.data_hash[tuple(params)])
        return []

    def AddColumn(self, column_data, column_names):
        if len(column_names) == 0:
            column_names = ["Col" + str(index) for index in range(len(column_data))]
        headers = self.headers_data + column_names
        new_data = [rdata + column_data for rdata in self.data_lines]
        return headers, new_data

    def WriteHeaderInCSV(self, filename, headers=None):
        self.csv_filename = filename
        print("[Info] Writing into file:"+filename)
        if headers == None:
            headers = self.headers_data
        with open(self.csv_filename, "w") as csv_fh:
            csv_fh.write(", ".join(headers) + "\n")
            csv_fh.close()

    def WriteDataInCSV(self, data):
        with open(self.csv_filename, "a") as csv_fh:
            csv_fh.writelines(", ".join(data) + "\n")
            csv_fh.close()

    def WriteExcel(self, filename, headers=None, data=None):
        xlsx_filename = filename + ".xlsx"
        csv_filename = filename + ".csv"
        if headers == None:
            headers = self.headers_data
        if data == None:
            data = self.data_lines
        print("Writing stats in output file:" + xlsx_filename)
        workbook = xlsxwriter.Workbook(xlsx_filename)
        worksheet = workbook.add_worksheet("Cycles")
        hdr_list = [{"header": hdr} for hdr in headers]
        cols = len(headers)
        rows = len(data)
        print("Headers:[" + ", ".join(headers) + "]")
        print("Records: " + str(len(data)))
        start_row = 0
        start_col = 0
        worksheet.add_table(
            start_row,
            start_col,
            start_row + rows - 1,
            start_col + cols - 1,
            {"data": data, "columns": hdr_list},
        )
        with open(csv_filename, "w") as fh:
            lines = [", ".join(map(str, data_row)) + "\n" for data_row in data]
            fh.write(", ".join(headers) + "\n")
            fh.writelines(lines)
            fh.close()
        workbook.close()


def StitchData(args):
    def GetDataAtIndexes(data, indexes):
        return [data[i] for i in indexes]

    input_list = []
    if args.inputs != "":
        input_list = args.inputs
    key_column_names = []
    if args.column_names != "":
        key_column_names = re.split(r"::", args.column_names)
    workloads = []
    for inp in input_list:
        fields = re.split(r"::", inp)
        filename = fields[-1]
        tag = fields[0]
        tag = re.sub(r"\.csv", "", tag)
        workload = Workload()
        workload.ReadPreEvaluatedWorkloads(filename)
        group_data = workload.GroupsData(key_column_names)
        valid_hdrs = workload.GetValidHdrs(key_column_names)
        workloads.append((workload, group_data, valid_hdrs, tag))
    headers_hash = {}
    ignore_fields = ["cpu", "WorkDir", "pid", "time", "interface"]
    map_hdrs = HeaderMap(args)
    for index, (workload, group_data, valid_hdrs, tag) in enumerate(workloads):
        hdr_indexes = workload.headers_index_hash
        for hdr in hdr_indexes.keys():
            lhdr = hdr
            if hdr in map_hdrs:
                lhdr = map_hdrs[hdr]
            if lhdr in ignore_fields:
                continue
            if lhdr not in headers_hash:
                headers_hash[lhdr] = {}
            headers_hash[lhdr][index] = hdr_indexes[hdr]
    all_hdrs = sorted(headers_hash.keys())
    all_common_hdrs = []
    for hdr in sorted(headers_hash.keys()):
        if len(headers_hash[hdr]) > 1:
            if hdr not in key_column_names:
                all_common_hdrs.append(hdr)
    group_data_keys = {}
    for index, (workload, group_data, valid_hdrs, tag) in enumerate(workloads):
        for k in group_data.keys():
            group_data_keys[k] = 1
    new_hdrs = []
    new_data = []
    new_hdrs.extend(key_column_names)
    for index, (workload, group_data, valid_hdrs, tag) in enumerate(workloads):
        for hdr in valid_hdrs:
            if hdr in all_common_hdrs:
                new_hdrs.append(tag + "-" + hdr)
            else:
                new_hdrs.append(hdr)
    for k in group_data_keys.keys():
        data_record = []
        if k == tuple:
            data_record.extend(list(k))
        else:
            data_record.append(k)
        for index, (workload, group_data, valid_hdrs, tag) in enumerate(workloads):
            indexes = [
                headers_hash[hdr][index] if index in headers_hash[hdr] else -1
                for hdr in valid_hdrs
            ]
            k_data = group_data[k]
            data_record.extend(GetDataAtIndexes(k_data[0], indexes))
        new_data.append(data_record)
    new_workload = Workload()
    new_workload.InitializeWorkload(new_hdrs, new_data)
    return new_workload


def CombineData(args):
    input_list = []
    if args.inputs != "":
        input_list = args.inputs
    column_names = []
    if args.column_names != "":
        column_names = re.split(r"::", args.column_names)
    workloads = []
    combined_data = []
    ignore_fields = ["cpu", "WorkDir", "pid", "time", "interface"]
    map_hdrs = HeaderMap(args)
    for inp in input_list:
        fields = re.split(r"::", inp)
        filename = fields[-1]
        workload = Workload()
        workload.ReadPreEvaluatedWorkloads(filename)
        workloads.append(workload)
    common_hdrs = {}
    for index, workload in enumerate(workloads):
        hdr_indexes = workload.headers_index_hash
        for hdr in hdr_indexes.keys():
            lhdr = hdr
            if hdr in map_hdrs:
                lhdr = map_hdrs[hdr]
            if lhdr in ignore_fields:
                continue
            if lhdr not in common_hdrs:
                common_hdrs[lhdr] = {}
            common_hdrs[lhdr][index] = hdr_indexes[hdr]
    if args.only_common_columns:
        for hdr in list(common_hdrs.keys()):
            for index, workload in enumerate(workloads):
                if index not in common_hdrs[hdr]:
                    del common_hdrs[hdr]
                    break
    hdrs = list(common_hdrs.keys())
    hdrs.sort()
    for index, workload in enumerate(workloads):
        inp = input_list[index]
        fields = re.split(r"::", inp)
        filename = fields[-1]
        add_columns = fields[:-1]
        indexes = [
            common_hdrs[hdr][index] if index in common_hdrs[hdr] else -1 for hdr in hdrs
        ]
        data = workload.Get2DDataWithCheck(indexes)
        data = [rdata + add_columns for rdata in data]
        combined_data = combined_data + data
    new_workload = Workload()
    new_workload.InitializeWorkload(hdrs + column_names, combined_data)
    # new_workload.WriteExcel(args.output, hdrs+column_names, combined_data)
    return new_workload


def HeaderMap(args):
    map_hdrs = {
        "l1da": "l1d_asc",
        "l1ia": "l1i_asc",
        "l1ds": "l1d_size",
        "l1is": "l1i_size",
        "l2a": "l2_asc",
        "l2s": "l2_size",
        "options-rows": "options",
        "cls": "cacheline",
        "%eval_id": "Index",
        "obj_fn_1": "cpu_cycles",
        "obj_fn_2": "System Leakage Power",
    }
    if args.map_hdrs != "":
        map_hdrs = {}
        for map_h in args.map_hdrs:
            fields = re.split(r"::", map_h)
            map_hdrs[fields[0]] = fields[1]
    return map_hdrs


def MergeWorkloads(args, workloads):
    map_hdrs = HeaderMap(args)
    combined_data = []
    common_hdrs = {}
    for index, workload in enumerate(workloads):
        hdr_indexes = workload.headers_index_hash
        for hdr in hdr_indexes.keys():
            lhdr = hdr
            if hdr in map_hdrs:
                lhdr = map_hdrs[hdr]
            if lhdr not in common_hdrs:
                common_hdrs[lhdr] = {}
            common_hdrs[lhdr][index] = hdr_indexes[hdr]
    if args.only_common_columns:
        for hdr in list(common_hdrs.keys()):
            for index, workload in enumerate(workloads):
                if index not in common_hdrs[hdr]:
                    del common_hdrs[hdr]
                    break
    hdrs = list(common_hdrs.keys())
    hdrs.sort()
    for index, workload in enumerate(workloads):
        indexes = [
            common_hdrs[hdr][index] if index in common_hdrs[hdr] else -1 for hdr in hdrs
        ]
        data = workload.Get2DDataWithCheck(indexes)
        combined_data = combined_data + data
    new_workload = Workload()
    new_workload.InitializeWorkload(hdrs, combined_data)
    # new_workload.WriteExcel(args.output, hdrs, combined_data)
    return new_workload


def MergeData(args):
    if args.inputs != "":
        input_list = args.inputs
    workloads = []
    for inp in input_list:
        filename = inp
        workload = Workload()
        workload.ReadPreEvaluatedWorkloads(filename)
        workloads.append(workload)
    new_workload = MergeWorkloads(args, workloads)
    return new_workload


def GetParetoData(xydata, deviation=0.0):
    xdata = np.array(xydata[0]).astype("float")
    ydata = np.array(xydata[1]).astype("float")
    best_point = [xdata[0], ydata[0]]
    prev_best_point = best_point
    pareto_point = [[], []]
    for index in range(xdata.size):
        if xdata[index] == best_point[0]:
            if ydata[index] < best_point[1]:
                best_point[1] = ydata[index]
        else:
            is_best = False
            is_second_best = False
            if len(pareto_point[1]) == 0 or best_point[1] < prev_best_point[1]:
                is_best = True
            if len(pareto_point[1]) == 0 or best_point[1] < (
                prev_best_point[1] + prev_best_point[1] * deviation
            ):
                is_second_best = True
            if is_best or is_second_best:
                pareto_point[0].append(best_point[0])
                pareto_point[1].append(best_point[1])
            if is_best:
                prev_best_point = best_point
            best_point = [xdata[index], ydata[index]]
    is_best = False
    is_second_best = False
    if len(pareto_point[1]) == 0 or best_point[1] < prev_best_point[1]:
        is_best = True
    if len(pareto_point[1]) == 0 or best_point[1] < (
        prev_best_point[1] + prev_best_point[1] * deviation
    ):
        is_second_best = True
    if is_best or is_second_best:
        pareto_point[0].append(best_point[0])
        pareto_point[1].append(best_point[1])
    print("Total pareto points: " + str(len(pareto_point[0])))
    return pareto_point


def GroupDataPlot(args, workload):
    # -group-plot Step::TestLoss
    workload = workload.SortWorkload(field_index=args.xcol, dtype=args.xdatatype)
    groups_axis = []
    if re.search("::", args.group_plot):
        groups_axis = re.split("\s*::\s*", args.group_plot)
    elif re.search(",", args.group_plot):
        groups_axis = re.split("\s*,\s*", args.group_plot)
    elif args.group_plot != "":
        groups_axis.append(args.group_plot)
    y_axis_index = workload.GetHdrIndex(args.ycol[0])
    min_max_std_indexes = []
    if len(args.ycol)>1:
        min_max_std_indexes = [workload.GetHdrIndex(y) 
                                         for y in args.ycol[1:]]
    x_axis_index = workload.GetHdrIndex(args.xcol)
    data_lines = workload.data_lines
    if args.xsort:
        data_lines = workload.SortData(data_lines, int, x_axis_index)
    if args.ysort:
        data_lines = workload.SortData(data_lines, float, y_axis_index)
    group_data = workload.GroupsData(groups_axis, data_lines)
    PlotGraph(args, group_data, 
            x_axis_index, y_axis_index, min_max_std_indexes)


def MultiDataPlot(args, workload):
    # -group-plot Step::TestLoss
    workload = workload.SortWorkload(field_index=args.xcol, dtype=args.xdatatype)
    data_lines = workload.data_lines
    if args.xsort:
        data_lines = workload.SortData(data_lines, int, x_axis_index)
    if args.ysort:
        data_lines = workload.SortData(data_lines, float, y_axis_index)
    ycols = []
    if args.ycol != "":
        ycols = args.ycol
    x_axis_index = workload.GetHdrIndex(args.xcol)
    group_data = {}
    for ycol in ycols:
        y_axis_index = workload.GetHdrIndex(ycol)
        data = workload.Get2DData([x_axis_index, y_axis_index], data_lines)
        group_data[ycol] = data
    PlotGraph(args, group_data, 0, 1, [])


def PlotGraph(args, group_data, x_axis_index, 
        y_axis_index, min_max_std_indexes):
    l_xmin = ""
    l_ymin = ""
    l_xmax = ""
    l_ymax = ""
    if args.xlimit != "":
        mm_range = re.split(":", args.xlimit)
        if len(mm_range) == 1:
            l_xmin = mm_range[0]
        if len(mm_range) == 2:
            l_xmin = mm_range[0]
            l_xmax = mm_range[1]
    if args.ylimit != "":
        mm_range = re.split(":", args.ylimit)
        if len(mm_range) == 1:
            l_ymin = mm_range[0]
        if len(mm_range) == 2:
            l_ymin = mm_range[0]
            l_ymax = mm_range[1]
    if args.plot_font_size != "":
        plt.rcParams.update({"font.size": int(args.plot_font_size)})

    def GetColumn(data, cols, min_max_std_indexes):
        def GetDataAtIndexes(data, indexes):
            return [data[i] for i in indexes]

        def IsValidColumn(data_line, cols):
            data_at_indexes = GetDataAtIndexes(data_line, cols)
            if l_xmin != "" and float(data_at_indexes[0]) < float(l_xmin):
                return False
            if l_ymin != "" and float(data_at_indexes[1]) < float(l_ymin):
                return False
            if l_xmax != "" and float(data_at_indexes[0]) > float(l_xmax):
                return False
            if l_ymax != "" and float(data_at_indexes[1]) > float(l_ymax):
                return False
            if "1e+18" in data_at_indexes:
                return False
            return True

        pruned_data = [
            data_line for data_line in data if IsValidColumn(data_line, cols)
        ]
        if len(pruned_data) == 0:
            return [[], []]
        d = zip(*pruned_data)
        d2 = list(d)
        xydata = [list(d2[col]) for col in cols]
        min_max_std_data = [list(d2[col]) for col in min_max_std_indexes]
        min_max_std_data = np.array(min_max_std_data).astype("float")
        if args.xdatatype == "" and IsFloat(xydata[0][0]):
            xydata0_np = np.array(xydata[0]).astype("float")
            xydata = [xydata0_np.tolist(), xydata[1]]
        if args.ydatatype == "" and IsFloat(xydata[1][0]):
            xydata1_np = np.array(xydata[1]).astype("float")
            xydata = [xydata[0], xydata1_np.tolist()]
        if args.xdatatype != "":
            xydata0_np = np.array(xydata[0]).astype(args.xdatatype)
            xydata = [xydata0_np.tolist(), xydata[1]]
        if args.ydatatype != "":
            xydata1_np = np.array(xydata[1]).astype(args.ydatatype)
            xydata = [xydata[0], xydata1_np.tolist()]
        return xydata, min_max_std_data

    def NormalizeData(x, xmin, xnorm):
        return (x - xmin) / xnorm

    markers = ["o", "x", "+", "v", "^", "<", ">", "s", "d", ".", "1", "2"]
    title = args.title
    xtitle = args.xtitle
    ytitle = args.ytitle
    ztitle = args.ztitle
    if xtitle == "":
        xtitle = "X"
    if ytitle == "":
        ytitle = "Y"
    if ztitle == "":
        ztitle = "Z"
    group_keys = []
    if args.group_keys != "":
        group_keys = re.split("::", args.group_keys)
    ymin = 1e19
    xmin = 1e19
    ymax = -1e-19
    xmax = -1e-19
    graph_data = {}
    group_key_hash = {}
    for key in group_keys:
        key_tuple = key
        if re.search(",", key):
            key_tuple = tuple(re.split("\s*,\s*", key))
        elif int_re.search(key):
            key_tuple = int(key)
        elif float_re.search(key):
            key_tuple = float(key)
        group_key_hash[key_tuple] = 1
    for key, ldata in group_data.items():
        xydata, min_max_std_data = GetColumn(ldata, [x_axis_index, y_axis_index], min_max_std_indexes)
        if len(xydata[0]) == 0:
            continue
        if args.pareto:
            xydata = GetParetoData(xydata, float(args.deviation))
        if type(key) != tuple:
            if int_re.search(key):
                key = int(key)
            elif float_re.search(key):
                key = float(key)
        graph_data[key] = (xydata, min_max_std_data)
        if args.plot_normalize:
            xmax = max(xmax, max(xydata[0]))
            xmin = min(xmin, min(xydata[0]))
            ymax = max(ymax, max(xydata[1]))
            ymin = min(ymin, min(xydata[1]))
    ax = None
    if args.plot3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    color_index = 0
    index = 0
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.2)
    xtick_labels_hash = {}
    xtick_labels = []
    total_xlabels = 0
    total_keys = len(graph_data.keys())
    xtick_args = {}
    xdatatype = args.xdatatype
    for key in sorted(graph_data.keys()):
        if len(group_key_hash) > 0 and key not in group_key_hash:
            continue
        (xydata, min_max_std_data) = graph_data[key]
        print("Identified group key:" + str(key) + " count:" + str(len(xydata[0])))
        for data in xydata[0]:
            if data not in xtick_labels_hash:
                xtick_labels_hash[data] = len(xtick_labels)
                xtick_labels.append(data)
        total_xlabels = max(total_xlabels, len(xtick_labels))
        x = np.array(xydata[0])
        y = np.array(xydata[1])
        if not IsFloat(x[0]):
            xdatatype = 'str'
        if args.plot_normalize:
            xnorm = xmax - xmin
            if xnorm == 0.0:
                xnorm = np.finfo(x.dtype).eps
            ynorm = ymax - ymin
            if ynorm == 0.0:
                ynorm = np.finfo(y.dtype).eps
            x = NormalizeData(x, xmin, xnorm)
            y = NormalizeData(y, ymin, ynorm)
        marker = markers[index % len(markers)]
        if ax != None:
            ax.bar(
                x,
                y,
                zs=key,
                width=100.0,
                zdir="y",
                color=colors[color_index % len(colors)],
                alpha=0.8,
            )
            color_index = color_index + 1
        elif args.min_max_std_plot:
            plt_color = colors[color_index%len(colors)]
            xfmt = x
            if xdatatype == 'str':
                xfmt = np.array([ xtick_labels_hash[d] for d in x ])
            if total_keys != 1:
                tick_range = 0.25
                xfmt = xfmt - tick_range + (2*tick_range*index/(total_keys-1))
            if len(min_max_std_indexes) > 2:
                plt.errorbar(xfmt, y, min_max_std_data[2], fmt=plt_color+marker, ecolor=plt_color, lw=3)
            mins = min_max_std_data[0]
            maxes = min_max_std_data[1]
            plt.errorbar(xfmt, y, [y-mins, maxes-y], fmt=plt_color+marker, ecolor=plt_color, lw=1)
            plt.plot(xfmt, y, marker, color=plt_color, linewidth="1", linestyle="-", label=str(key))
            color_index = color_index + 1
        else:
            plt.plot(x, y, marker, linewidth="1", linestyle="-", label=str(key))
        index = index + 1
    if ax != None:
        ax.set_label(xtitle)
        ax.set_label(ytitle)
        ax.set_label(ztitle)
    else:
        plot_width = re.split(r":", args.plot_width_loc)
        if len(plot_width) == 1:
            plot_width.append(plot_width[0])
            plot_width[0] = 0
        plot_height = re.split(r":", args.plot_height_loc)
        if len(plot_height) == 1:
            plot_height.append(plot_height[0])
            plot_height[0] = 0
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0 + box.width * float(plot_width[0]), 
                box.y0 + box.height * float(plot_height[0]),
             box.width * float(plot_width[1]), box.height * float(plot_height[1])])
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.legend(numpoints=1)
        legend_args = {}
        if args.legend_title != "":
            legend_args['title'] = args.legend_title
        if args.legend_pos != "":
            legend_pos = args.legend_pos
            legend_pos = re.sub(r":", " ", legend_pos)
            legend_args['loc']=legend_pos
        if args.legend_ncol != '':
            legend_args['ncol'] = int(args.legend_ncol)
        if args.legend_fontsize != '':
            legend_args['fontsize'] = int(args.legend_fontsize)
        if args.legend_bbox != '':
            bbox = args.legend_bbox
            bbox = re.split(r"\s*[,:]\s*", bbox)
            fbox = [float(x) for x in bbox]
            fbox = tuple(fbox)
            legend_args['bbox_to_anchor'] = fbox
        plt.legend(**legend_args)
    if xdatatype == "str":
        xtick_args["ticks"] = np.arange(total_xlabels)
        xtick_args["labels"] = xtick_labels
    if args.xticks_rotation != "":
        xtick_args["rotation"] = int(args.xticks_rotation)
    plt.xticks(**xtick_args)
    if args.yticks_rotation != "":
        plt.yticks(rotation=int(args.yticks_rotation))
    if title != "" and not args.notitle:
        plt.title(title)
    if args.log != "":
        plt.yscale("log", basey=int(args.log))
    if args.ylog != "":
        plt.yscale("log", basey=int(args.ylog))
    if args.xlog != "":
        plt.xscale("log", basex=int(args.xlog))
    print("Writing the output plot:" + args.plot_name)
    plt.savefig(args.plot_name)


def GetNormalizedData(workload):
    data = workload.Get2DDataWithNormalization(
        workload.param_indexes + workload.cost_indexes, workload.GetData(), {}
    )
    hdrs = workload.GetParamCostHeaders()
    workload = Workload()
    workload.InitializeWorkload(hdrs, data)
    return workload


def GetSequencing(args, workload):
    second_indexing = workload.GetHdrIndex(args.sequencing[1])
    key_hash = {}
    group_data = workload.GroupsData([args.sequencing[0]])
    keys = sorted(group_data.keys(), key=int)
    count_data = int(args.sequence_count)
    new_data = []
    prev_max_sk = 0
    for k in keys:
        data = group_data[k]
        sub_group = workload.GroupsData([args.sequencing[1]], data)
        sub_keys = sorted(sub_group.keys(), key=int)
        max_sk = prev_max_sk
        for sk in sub_keys:
            new_sk = prev_max_sk + int(sk)
            recs = sub_group[sk]
            if new_sk > max_sk:
                max_sk = new_sk
            for rec in recs:
                rec[second_indexing] = new_sk
                new_data.append(rec)
        if count_data == -1:
            prev_max_sk = max_sk
        else:
            prev_max_sk = max(prev_max_sk + count_data, max_sk)
    hdrs = workload.GetHeaders()
    workload = Workload()
    workload.InitializeWorkload(hdrs, new_data)
    return workload


def GetJustSufficientData(workload):
    data = workload.Get2DParametersCostData()
    hdrs = workload.GetParamCostHeaders()
    workload = Workload()
    workload.InitializeWorkload(hdrs, data)
    return workload


def GetSimpleConditionWorkload(args, workload, condition):
    const_re = re.compile(r"^\s*::\s*(.*)\s*$", re.IGNORECASE)
    fields_re = re.compile(r"\s*([^\s=<>]*)\s*(==|<=|>=|<|>)\s*(.*)\s*", re.IGNORECASE)
    fields = fields_re.search(condition)
    if fields:
        data = workload.GetData()
        hdrs = workload.GetHeaders()
        src0_index = workload.GetHdrIndex(fields[1])
        operator = fields[2]
        const_fields = const_re.search(fields[3])
        if const_fields:
            src1_index = const_fields[1]
        else:
            src1_index = workload.GetHdrIndex(fields[3])
        new_data = None
        if operator == "==":
            if const_fields:
                new_data = [rdata for rdata in data if rdata[src0_index] == src1_index]
            else:
                new_data = [
                    rdata for rdata in data if rdata[src0_index] == rdata[src1_index]
                ]
        elif operator == "<=":
            if const_fields:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) <= float(src1_index)
                ]
            else:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) <= float(rdata[src1_index])
                ]
        elif operator == "<":
            if const_fields:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) < float(src1_index)
                ]
            else:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) < float(rdata[src1_index])
                ]
        elif operator == ">=":
            if const_fields:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) >= float(src1_index)
                ]
            else:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) >= float(rdata[src1_index])
                ]
        elif operator == ">":
            if const_fields:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) > float(src1_index)
                ]
            else:
                new_data = [
                    rdata
                    for rdata in data
                    if float(rdata[src0_index]) > float(rdata[src1_index])
                ]
        workload = Workload()
        workload.InitializeWorkload(hdrs, new_data)
    return workload


def PerformCondition(args, workload):
    condition = args.condition
    comb_re = re.compile(r"&&|\|\|", re.IGNORECASE)
    if comb_re.search(condition):
        sp = re.split(r"\s*(&&|\|\|)\s*", condition)
        for index in range(1, len(sp), 2):
            if sp[index] == "&&":
                print(
                    "Logic condition on : -{}- and -{}-".format(
                        sp[index - 1], sp[index + 1]
                    )
                )
                workload = GetSimpleConditionWorkload(args, workload, sp[index - 1])
                workload = GetSimpleConditionWorkload(args, workload, sp[index + 1])
            elif sp[index] == "||":
                print(
                    "Logic condition on : {} or {}".format(sp[index - 1], sp[index + 1])
                )
                workload1 = GetSimpleConditionWorkload(args, workload, sp[index - 1])
                workload2 = GetSimpleConditionWorkload(args, workload, sp[index + 1])
                workload = MergeWorkloads(args, [workload1, workload2])
    else:
        workload = GetSimpleConditionWorkload(args, workload, condition)
    return workload


def PerformStats(args, workload):
    const_re = re.compile(r"^\s*::\s*(.*)\s*$", re.IGNORECASE)
    fields_re = re.compile(
        r"\s*([^=]*)\s*=\s*([^\s]*)\s\s*([^\s]*)\s\s*([^\s]*)", re.IGNORECASE
    )
    fields = fields_re.search(args.stats)
    if fields:
        data = workload.GetData()
        hdrs = workload.GetHeaders()
        if not workload.IsHdrExist(re.sub("\s*$", "", fields[1])):
            dst_index = len(hdrs)
            hdrs, data = workload.AddColumn(["0"], [re.sub("\s*$", "", fields[1])])
        else:
            dst_index = workload.GetHdrIndex(re.sub("\s*$", "", fields[1]))
        src0_index = workload.GetHdrIndex(re.sub("\s*$", "", fields[2]))
        operator = re.sub("\s*$", "", fields[3])
        const_fields = const_re.search(fields[4])
        if const_fields:
            src1_index = const_fields[1]
        else:
            src1_index = workload.GetHdrIndex(re.sub("\s*$", "", fields[4]))
        if operator.lower() == "min":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(min(
                                float(rdata[src0_index]), 
                                float(src1_index)
                                ))
            else:
                for rdata in data:
                    rdata[dst_index] = str(min(
                        float(rdata[src0_index]), 
                        float(rdata[src1_index])
                        ))
        elif operator == "max":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(max(
                                float(rdata[src0_index]), 
                                float(src1_index)
                                ))
            else:
                for rdata in data:
                    rdata[dst_index] = str(max(
                        float(rdata[src0_index]), 
                        float(rdata[src1_index])
                    ))
        elif operator == "avg":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(np.mean([
                                float(rdata[src0_index]), 
                                float(src1_index)
                    ]))
            else:
                for rdata in data:
                    rdata[dst_index] = str(np.mean([
                        float(rdata[src0_index]),
                        float(rdata[src1_index])
                    ]))
        workload = Workload()
        workload.InitializeWorkload(hdrs, data)
    return workload

def PerformArithmatic(args, workload):
    const_re = re.compile(r"^\s*::\s*(.*)\s*$", re.IGNORECASE)
    fields_re = re.compile(
        r"\s*([^=]*)\s*=\s*([^\+\-\*\/]*)\s*([\+\-\*\/])\s*(.*)", re.IGNORECASE
    )
    fields = fields_re.search(args.arith)
    if fields:
        data = workload.GetData()
        hdrs = workload.GetHeaders()
        if not workload.IsHdrExist(re.sub("\s*$", "", fields[1])):
            dst_index = len(hdrs)
            hdrs, data = workload.AddColumn(["0"], [re.sub("\s*$", "", fields[1])])
        else:
            dst_index = workload.GetHdrIndex(re.sub("\s*$", "", fields[1]))
        src0_index = workload.GetHdrIndex(re.sub("\s*$", "", fields[2]))
        operator = re.sub("\s*$", "", fields[3])
        const_fields = const_re.search(fields[4])
        if const_fields:
            src1_index = const_fields[1]
        else:
            src1_index = workload.GetHdrIndex(re.sub("\s*$", "", fields[4]))
        if operator == "+":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(float(rdata[src0_index]) + float(src1_index))
            else:
                for rdata in data:
                    rdata[dst_index] = str(
                        float(rdata[src0_index]) + float(rdata[src1_index])
                    )
        elif operator == "-":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(float(rdata[src0_index]) - float(src1_index))
            else:
                for rdata in data:
                    rdata[dst_index] = str(
                        float(rdata[src0_index]) - float(rdata[src1_index])
                    )
        elif operator == "*":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(float(rdata[src0_index]) * float(src1_index))
            else:
                for rdata in data:
                    rdata[dst_index] = str(
                        float(rdata[src0_index]) * float(rdata[src1_index])
                    )
        elif operator == "/":
            if const_fields:
                for rdata in data:
                    rdata[dst_index] = str(float(rdata[src0_index]) / float(src1_index))
            else:
                for rdata in data:
                    rdata[dst_index] = str(
                        float(rdata[src0_index]) / float(rdata[src1_index])
                    )
        workload = Workload()
        workload.InitializeWorkload(hdrs, data)
    return workload


def ProcessData(args):
    workload = None
    write_flag = False
    if args.combine:
        workload = CombineData(args)
        write_flag = True
    elif args.merge:
        workload = MergeData(args)
        write_flag = True
    elif args.stitch:
        workload = StitchData(args)
        write_flag = True
    else:
        workload = Workload()
        workload.ReadPreEvaluatedWorkloads(args.inputs[0])
    if args.arith != "":
        workload = PerformArithmatic(args, workload)
        write_flag = True
    if args.stats != "":
        workload = PerformStats(args, workload)
        write_flag = True
    if args.condition != "":
        workload = PerformCondition(args, workload)
        write_flag = True
    if args.just_sufficient:
        workload = GetJustSufficientData(workload)
        write_flag = True
    if args.normalize:
        workload = GetNormalizedData(workload)
        write_flag = True
    if args.sequencing != "":
        workload = GetSequencing(args, workload)
        write_flag = True
    if write_flag:
        workload.WriteExcel(args.output)
    if args.group_plot != "":
        GroupDataPlot(args, workload)
    elif args.multi_plot:
        MultiDataPlot(args, workload)


def InitializeWorkloadArgParse(parser):
    parser.add_argument("-map", nargs="*", action="store", dest="map_hdrs", default="")
    parser.add_argument("-input", nargs="*", action="store", dest="inputs", default="")
    parser.add_argument("-min-max-std-plot", dest="min_max_std_plot", action="store_true")
    parser.add_argument("-combine", dest="combine", action="store_true")
    parser.add_argument("-stitch", dest="stitch", action="store_true")
    parser.add_argument("-merge", dest="merge", action="store_true")
    parser.add_argument(
        "-common-columns", dest="only_common_columns", action="store_true"
    )
    parser.add_argument("-output", dest="output", default="output")
    parser.add_argument("-column-names", dest="column_names", default="")
    parser.add_argument("-multi-plot", dest="multi_plot", action="store_true")
    parser.add_argument("-plot-width", dest="plot_width_loc", default="0:1.0")
    parser.add_argument("-plot-height", dest="plot_height_loc", default="0:1.0")
    parser.add_argument("-group-plot", dest="group_plot", default="")
    parser.add_argument("-group-keys", dest="group_keys", default="")
    parser.add_argument("-sequence-count", dest="sequence_count", default="-1")
    parser.add_argument(
        "-sequencing", nargs="*", action="store", dest="sequencing", default=""
    )
    parser.add_argument("-plot3d", dest="plot3d", action="store_true")
    parser.add_argument("-normalize", dest="normalize", action="store_true")
    parser.add_argument("-arith", dest="arith", default="")
    parser.add_argument("-condition", dest="condition", default="")
    parser.add_argument("-stats", dest="stats", default="")
    parser.add_argument("-just-sufficient", dest="just_sufficient", action="store_true")
    parser.add_argument("-plot-font-size", dest="plot_font_size", default="")
    parser.add_argument("-plot-name", dest="plot_name", default="out_plot.png")
    parser.add_argument("-plot-normalize", dest="plot_normalize", action="store_true")
    parser.add_argument("-pareto", dest="pareto", action="store_true")
    parser.add_argument("-pareto-deviation", dest="deviation", default="0.0")
    parser.add_argument("-title", dest="title", default="")
    parser.add_argument("-notitle", dest="notitle", action="store_true")
    parser.add_argument("-ydatatype", dest="ydatatype", default="")
    parser.add_argument("-xdatatype", dest="xdatatype", default="")
    parser.add_argument("-legend-title", dest="legend_title", default="")
    parser.add_argument("-legend-pos", dest="legend_pos", default="best")
    parser.add_argument("-legend-ncol", dest="legend_ncol", default="")
    parser.add_argument("-legend-fontsize", dest="legend_fontsize", default="")
    parser.add_argument("-legend-bbox", dest="legend_bbox", default="")
    parser.add_argument("-xtitle", dest="xtitle", default="")
    parser.add_argument("-ytitle", dest="ytitle", default="")
    parser.add_argument("-ztitle", dest="ztitle", default="")
    parser.add_argument("-xticks-rotation", dest="xticks_rotation", default="")
    parser.add_argument("-yticks-rotation", dest="yticks_rotation", default="")
    parser.add_argument("-xlimit", dest="xlimit", default="")
    parser.add_argument("-ylimit", dest="ylimit", default="")
    parser.add_argument("-xcol", dest="xcol", default="Index")
    parser.add_argument(
        "-ycol", nargs="*", action="store", dest="ycol", default="cpu_cycles"
    )
    parser.add_argument("-xsort", dest="xsort", action="store_true")
    parser.add_argument("-ysort", dest="ysort", action="store_true")
    parser.add_argument("-zcol", dest="zcol", default="Z")
    parser.add_argument("-log", dest="log", default="")
    parser.add_argument("-ylog", dest="ylog", default="")
    parser.add_argument("-xlog", dest="xlog", default="")


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    InitializeWorkloadArgParse(parser)
    args = parser.parse_args()
    ProcessData(args)
    lapsed_time = "{:.3f} seconds".format(time.time() - start)
    print("Total runtime of script: " + lapsed_time)


if __name__ == "__main__":
    main()
