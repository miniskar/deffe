#! /bin/bash

## post-processing script 
## it should parse the results and generate a file named "results.out"
## for the case with only one objective function, it should have only one line,
## indicating the objective function

#python extract_single_test_data.py  | sed 's/:/ /g' | awk '{if( NR==1 || NR==2 || NR==3 || NR==4 ){print $NF}}' > results.tmp
python extract_single_test_data.py  | sed 's/:/ /g' | awk '{if( NR==1 || NR==2 ){print $NF}}' > results.tmp

mv results.tmp $1

## the current version only copies the first entry, the CPU cycles
