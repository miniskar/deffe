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
import os
import multiprocessing
from multiprocessing.pool import ThreadPool
import subprocess
import time
import pdb
from deffe_utils import *

class MultiThreadBatchRun:
    def __init__(self, taskpool_size, framework=None):
        self.pool = ThreadPool(taskpool_size)
        self.results = []
        self.index = 0
        self.framework = framework
        None

    def RunCommand(self, index, cmd, callback=None):
        # os.system(cmd)
        #print("Received command: "+str(cmd))
        start = time.time()
        os_cmd = cmd
        call_back_args = ()
        if type(cmd)==tuple:
            os_cmd = cmd[0]
            call_back_args = tuple(list(cmd)[1:])
        LogCmd(str(index) +": "+ os_cmd)
        os.system(os_cmd)
        end = time.time()
        self.results[index] = 1
        lapsed_time = "{:.3f} seconds".format(time.time() - start)
        Log(f"Index:{index} Lapsed time:" + str(lapsed_time))
        if callback != None:
            other_callbacks = callback[0:1]+callback[2:]+call_back_args
            callback[1](*other_callbacks)

    def Run(self, cmds, is_popen=False, callback=None):
        # print("Start running commands now:"+str(is_popen))
        for index, cmd in enumerate(cmds):
            self.results.append(0)
            if self.framework == None or not self.framework.args.no_run:
                self.pool.apply_async(self.RunCommand, (self.index, cmd, callback))
            else:
                os_cmd = cmd
                call_back_args = ()
                if type(cmd)==tuple:
                    os_cmd = cmd[0]
                    call_back_args = tuple(list(cmd)[1:])
                LogCmd(str(self.index) +": "+ os_cmd)
                if callback != None:
                    other_callbacks = callback[0:1]+callback[2:]+call_back_args
                    callback[1](*other_callbacks)
            self.index = self.index + 1
        # print("Jobs submitted")

    def Clear(self):
        self.index = 0
        del self.results[:]

    def GetActiveCount(self):
        sum = 0
        for res in self.results:
            sum = sum + res
        return sum

    def Join(self):
        self.pool.join()

    def Close(self):
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    cmds = ["sh $DEFFE_DIR/utils/samp.sh" for i in range(10)]
    mt = MultiThreadBatchRun(5)
    mt.Run(cmds, is_popen=True)
    mt.Run(cmds)
    for i in range(1000):
        time.sleep(1)
        print(mt.GetActiveCount())
    print("Waiting for jobs to be completed")
    mt.Close()
