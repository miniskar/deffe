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
import pdb

class DeffeSlurm:
    def __init__(self, framework, slurm_config=None):
        self.framework = framework
        if slurm_config  != None:
            self.config = slurm_config
        else:
            self.config = self.framework.config.GetSlurm()
        self.nodes = self.config.nodes
        self.cpus_per_task = self.config.cpus_per_task
        self.mem = self.config.mem
        self.mail_type = self.config.mail_type
        self.mail_user = self.config.mail_user
        self.account = self.config.account
        self.time = self.config.time
        self.exclude = self.config.exclude
        self.nodelist = self.config.nodelist
        self.constraint = self.config.constraint
        self.partition = self.config.partition
        self.source_scripts = self.config.source_scripts

    def CreateSlurmScript(self, cmd, slurm_filename):
        with open(slurm_filename, "w") as fh:
            fh.write("#!/bin/bash\n")
            if self.nodes != '':
                fh.write("#SBATCH --nodes=" + self.nodes + "\n")
            if self.cpus_per_task != '':
                fh.write("#SBATCH --cpus-per-task=" + self.cpus_per_task + "\n")
            if self.exclude != '':
                fh.write("#SBATCH --exclude='" + self.exclude+ "'\n")
            if self.nodelist != '':
                fh.write("#SBATCH --nodelist='" + self.nodelist+ "'\n")
            if self.constraint != '':
                fh.write('#SBATCH --constraint="' + self.constraint + '"\n')
            if self.partition != '':
                fh.write('#SBATCH --partition="' + self.partition+ '"\n')
            if self.mem != '':
                fh.write('#SBATCH --mem='+self.mem+"\n")
            if self.mail_type != '':
                fh.write('#SBATCH --mail-type='+self.mail_type+"\n")
            if self.mail_user != '':
                fh.write('#SBATCH --mail-user='+self.mail_user+"\n")
            if self.account != '':
                fh.write('#SBATCH --account='+self.account+"\n")
            if self.time != '':
                fh.write('#SBATCH --time='+self.time+"\n")
            fh.write("set -x;\n")
            fh.write('echo "Running on host: `hostname`"\n')
            fh.write('echo "SLURM_JOB_ID: $SLURM_JOB_ID"\n')
            fh.write('echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"\n')
            for sc in self.source_scripts:
                fh.write("source "+sc+"\n")
            fh.write("cd " + os.path.dirname(os.path.abspath(slurm_filename)) + " ; \n")
            #fh.write('echo "' + cmd + '"\n')
            fh.write(cmd + "\n")
            fh.write("cd -\n")
            fh.write('echo "Completed job!"\n')
            fh.close()

    def GetSlurmJobCommand(self, slurm_filename):
        return "sbatch -W " + slurm_filename + " ; wait"


def GetObject(*args):
    obj = DeffeSlurm(*args)
    return obj
