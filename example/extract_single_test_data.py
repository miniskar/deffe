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
import re
import os
import argparse

"""
 Usage: python extract_single_test_data.py -cwd \
   run_kmeans/l1da_4_l2s_128kB_options_kdd_cup-100_l2a_1_l1ia_1_l1ds_16kB_cls_32_l1is_1kB_cpu_TimingSimpleCPU
"""

stats = {"system.cpu.numCycles": "cpu_cycles"}
required_mcpat = [
    ("Processor", "Peak Power", "System Peak Power"),
    ("Processor", "Total Leakage", "System Leakage Power"),
    ("Core", "Peak Dynamic", "Core DPower"),
    ("Processor", "Area", "System Area"),
    ("Core", "Area", "Core Area"),
    ("Load Store Unit", "Area", "LSU Area"),
    ("Load Store Unit", "Peak Dynamic", "LSU DPower"),
    ("Execution Unit", "Area", "Ex Area"),
    ("Execution Unit", "Peak Dynamic", "Ex DPower"),
    ("Memory Controller", "Area", "Ex Area"),
    ("Memory Controller", "Peak Dynamic", "Ex DPower"),
]

# Get yosys file
def GetYosysFile(cwd, full_path=True):
    yosys_file = "yosys-stats.txt"
    if full_path:
        yosys_file = os.path.join(cwd, yosys_file)
    return yosys_file


# Get stats file
def GetStatsFile(cwd, full_path=True):
    stats_file = os.path.join("m5out", "stats.txt")
    if full_path:
        stats_file = os.path.join(cwd, stats_file)
    return stats_file


# Get mcpat file
def GetMCPatFile(cwd, full_path=True):
    mcpath_file = "mcpat.txt"
    if full_path:
        mcpath_file = os.path.join(cwd, mcpath_file)
    return mcpath_file


# Remove whitespaces in line
def RemoveWhiteSpaces(line):
    line = re.sub(r"\r", "", line)
    line = re.sub(r"\n", "", line)
    line = re.sub(r"^\s*$", "", line)
    line = re.sub(r"^\s*", "", line)
    line = re.sub(r"\s*$", "", line)
    return line


def ReadStatsFile(filename):
    data_hash = {}
    if not os.path.exists(filename):
        return data_hash
    with open(filename, "r") as fh:
        lines = fh.readlines()
        for line in lines:
            line = RemoveWhiteSpaces(line)
            if line == "":
                continue
            fields = re.split(r"\s+", line)
            if fields[0] in stats:
                target_field = stats[fields[0]]
                if fields[1].isdigit():
                    data_hash[target_field] = int(fields[1])
                else:
                    data_hash[target_field] = fields[1]
        fh.close()
    return data_hash


def ReadYosysResult(yosys_file):
    data_hash = {}
    if not os.path.exists(yosys_file):
        return data_hash
    # print("[Info] Reading yosys output in "+yosys_file)
    with open(yosys_file, "r") as fh:
        lines = [RemoveWhiteSpaces(line) for line in fh.readlines()]
        fh.close()
        comp = re.compile(r"^=== design hierarchy ===")
        number = re.compile(
            r"^\s*Number of (wires|wire bits|memories|memory bits|cells):\s*([^\s$]*)\s*$"
        )
        enabled = False
        for line in lines:
            if comp.search(line):
                enabled = True
            if enabled and number.search(line):
                key = number.group(1)
                value = number.group(1)
                data_hash[key] = value
    return data_hash


col_rx = re.compile("\s*:\s*", re.VERBOSE)
eq_rx = re.compile("\s*=\s*", re.VERBOSE)
float_rx = re.compile(
    "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?", re.VERBOSE
)


def ReadMCPatFile(filename):
    tags = {}
    current_tag = "main"
    tags[current_tag] = {}
    if not os.path.exists(filename):
        return tags
    with open(filename, "r") as fh:
        lines = fh.readlines()
        for line in lines:
            line = RemoveWhiteSpaces(line)
            if line == "":
                continue
            if col_rx.findall(line):
                fields = re.split(r"\s*:\s*", line)
                current_tag = fields[0]
                tags[current_tag] = {}
                continue
            if eq_rx.findall(line):
                fields = re.split(r"\s*=\s*", line)
                groups = float_rx.findall(fields[1])
                if groups:
                    tags[current_tag][fields[0]] = float(groups[0])
                else:
                    tags[current_tag][fields[0]] = fields[1]
                continue
    return tags


def GetData(cwd):
    stats_file = GetStatsFile(cwd, True)
    mcpat_file = GetMCPatFile(cwd, True)
    yosys_file = GetYosysFile(cwd, True)
    data_hash = ReadStatsFile(stats_file)
    mcpat_data = ReadMCPatFile(mcpat_file)
    yosys_data = ReadYosysResult(yosys_file)
    for (parent, child, dest) in required_mcpat:
        if parent in mcpat_data:
            if child in mcpat_data[parent]:
                data_hash[dest] = mcpat_data[parent][child]
    for k, v in yosys_data.items():
        data_hash[k] = v
    return data_hash


def GetProfileData(data_hash, key):
    if key not in data_hash:
        return pow(10, 18)
    return data_hash[key]


def main(args):
    data_hash = GetData(args.cwd)
    print("CPU cycles:" + str(GetProfileData(data_hash, "cpu_cycles")))
    print("Dynamic power:" + str(GetProfileData(data_hash, "Core DPower")))
    print("Cells:" + str(GetProfileData(data_hash, "cells")))
    print("Memory Bits:" + str(GetProfileData(data_hash, "memory bits")))
    print("System Peak Power:" + str(GetProfileData(data_hash, "System Peak Power")))
    print(
        "System Leakage Power:" + str(GetProfileData(data_hash, "System Leakage Power"))
    )
    print("System Area:" + str(GetProfileData(data_hash, "System Area")))
    print("Core Area:" + str(GetProfileData(data_hash, "Core Area")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cwd", dest="cwd", default=".")
    args = parser.parse_args()
    main(args)
