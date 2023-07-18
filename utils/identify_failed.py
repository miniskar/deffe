import re

def RemoveWhiteSpaces(line):
    line = re.sub(r"\r", "", line)
    line = re.sub(r"\n", "", line)
    line = re.sub(r"^\s*$", "", line)
    line = re.sub(r"^\s*", "", line)
    line = re.sub(r"\s*$", "", line)
    line = re.sub(r"kdd_cup-", "", line)
    return line

def GetSplitFields(line):
    re_comma_comp = re.compile(r",")
    line = RemoveWhiteSpaces(line)
    if line == "":
        return []
    data_fields_core = None
    if re_comma_comp.search(line):
        data_fields_core = re.split(r"\s*,\s*", line)
    else:
        data_fields_core = re.split(r"\s+", line)
    return data_fields_core

def read_file(csv_file):
    data = []
    with open(csv_file) as fh:
        all_lines = fh.readlines()
        if len(all_lines) == 0:
            return
        for line in all_lines:
            data.append(GetSplitFields(line))
        fh.close()
    return data

def identify_valid_experiments(data):
    if len(data) <= 1:
        return
    with open("valid_data.csv", "w") as fh:
        data_header = data[0]
        fh.write(", ".join(data_header)+"\n")
        for record in data[1:]:
            if len(record) == len(data_header):
                fh.write(", ".join(record)+"\n")
        fh.close()
def identify_failed_experiments(data):
    if len(data) <= 1:
        return
    with open("failed_data.csv", "w") as fh:
        data_header = data[0]
        fh.write(", ".join(data_header)+"\n")
        for record in data[1:]:
            if len(record) < len(data_header):
                fh.write(", ".join(record)+"\n")
        fh.close()

data = read_file("deffe_eval_predict.csv")
identify_failed_experiments(data)
identify_valid_experiments(data)
