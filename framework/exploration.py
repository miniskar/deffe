## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
 # @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
 #         miniskarnr@ornl.gov
 # 
 # Modification:
 #              Baseline code
 # Date:        Apr, 2020
 #**************************************************************************
###
import os
import argparse
import shlex

class DeffeExploration:
    def __init__(self, framework):
        self.config = framework.config.GetExploration()
        self.framework = framework

    # Initialize the members
    def Initialize(self):
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()

    # Read arguments provided in JSON configuration file
    def ReadArguments(self):
        arg_string = self.config.arguments
        args = self.parser.parse_args(shlex.split(arg_string))
        return args
    
    # Add command line arguments to parser
    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        return parser
    
    # Is exploration completed
    def IsCompleted(self):
        return self.framework.sampling.IsCompleted()

    # Returns true if model is ready 
    def IsModelReady(self):
        if self.framework.IsModelReady():
            return True
        return False

    def Run(self):
        None

def GetObject(framework):
    obj = DeffeExploration(framework)
    return obj
