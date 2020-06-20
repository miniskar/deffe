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

framework_path = os.getenv("DEFFE_DIR")
if framework_path == None:
    framework_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.environ["DEFFE_DIR"] = framework_path
print("Deffe framework is found in path: "+os.getenv("DEFFE_DIR"))
sys.path.insert(0, os.getenv("DEFFE_DIR"))
sys.path.insert(0, os.path.join(framework_path, "utils"))
sys.path.insert(0, os.path.join(framework_path, "ml_models"))
sys.path.insert(0, os.path.join(framework_path, "framework"))
from framework import *

# Main function
def main():
    framework = DeffeFramework()
    framework.Initialize()
    framework.Run()

if __name__ == "__main__":
    print("Current directory: " + os.getcwd())
    print("Machine: " + socket.gethostname())
    start = time.time()
    main()
    lapsed_time = "{:.3f} seconds".format(time.time() - start)
    print("Total runtime of script: " + lapsed_time)
