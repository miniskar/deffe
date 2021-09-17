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


class DeffeSlurm:
    def __init__(self, framework):
        self.framework = framework
        self.config = self.framework.config.GetSlurm()
        self.nodes = self.config.nodes
        self.cpus_per_task = self.config.cpus_per_task
        self.mem = self.config.mem
        self.constraint = self.config.constraint

    def CreateSlurmScript(self, cmd, slurm_filename):
        with open(slurm_filename, "w") as fh:
            fh.write("#!/bin/bash\n")
            if self.nodes != '':
                fh.write("#SBATCH --nodes=" + self.nodes + "\n")
            if self.cpus_per_task != '':
                fh.write("#SBATCH --cpus-per-task=" + self.cpus_per_task + "\n")
            fh.write('#SBATCH --constraint="' + self.constraint + '"\n')
            if self.mem != '':
                fh.write('#SBATCH --mem='+self.mem+"\n")
            fh.write('echo "Running on host: `hostname`"\n')
            fh.write('echo "SLURM_JOB_ID: $SLURM_JOB_ID"\n')
            fh.write('echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"\n')
            fh.write("cd " + os.path.dirname(os.path.abspath(slurm_filename)) + " ; \n")
            fh.write('echo "' + cmd + '"\n')
            fh.write(cmd + "\n")
            fh.write("cd -\n")
            fh.write('echo "Completed job!"\n')
            fh.close()

    def GetSlurmJobCommand(self, slurm_filename):
        return "sbatch -W " + slurm_filename + " ; wait"


def GetObject(framework):
    obj = DeffeSlurm(framework)
    return obj
