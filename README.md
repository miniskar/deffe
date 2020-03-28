This repository has the sources for Deffe framework, which is intended for design space exploration with exploration and machine learning based prediction capabilities. The state of the art design space exploration tools evaluate the design points (samples) and identify the optimal design points based on the cost metrics. However, the evaluation of design points are time consuming and may require heavy computation for some problems. Deffe will help for such problems. 

Deffe's machine learning model tries to learn from the evaluated design points. Once after learning for some time, Deffe's machine learning model inference can predict the cost metrics with ~98% accuracy, which can be used for fast design space exploration. It needs less than 5% of the samples to get the near-accurate machine learning prediction model and can save huge time in the design space exploration. 

Deffe framework is implemented fully in python and it is configured through json file. The json file has configuration of problem with parameters, cost metrics, evaluate procedure, prediction method, sampling technique and exploration algorithm. It provides great flexibility to users to add their custom python modules for the above tasks. 

## Deffe framework component diagram
![header image](docs/deffe-block-diagram.svg)

The above figures shows the main blocks and their corresponding python files in the Deffe framework.

## How to run Deffe?
An example Deffe configuration for RISCV design space exploration along with their associated files are placed in <b>example</b> directory. 
```
$ source setup.source
$ cd example ; 
$ python3 ../framework/run_deffe.py -config config_small.json
$ cd .. ;
```

The run_deffe.py file can show all command line options with the below command.
```
$ cd example ;
$ python3 ../framework/run_deffe.py -h
$ cd .. ;
```

To run the exploration on the preloaded data
```
$ cd example ;
$ python3 ../framework/run_deffe.py -config config_small.json -step-start 0 -step-end 1 -epochs 100 -batch-size 256 -only-preloaded-data-exploration
$ cd .. ;
```
