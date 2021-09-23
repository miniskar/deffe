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
import glob, os, sys
import socket
import pdb
import signal

from framework import *

# Main function
def main():
    config_data = None
    framework = DeffeFramework()
    framework.InitParser()
    framework.ReadArguments()
    framework.Initialize(config_data)
    framework.Run()

if __name__ == "__main__":
    print("Current directory: " + os.getcwd())
    print("Machine: " + socket.gethostname())
    start = time.time()
    main()
    lapsed_time = "{:.3f} seconds".format(time.time() - start)
    print("Total runtime of script: " + lapsed_time)
