import os


class DeffeSlurm:
    def __init__(self, framework):
        self.framework = framework
        self.config = self.framework.config.GetSlurm()
        self.nodes = self.config.nodes
        self.cpus_per_task = self.config.cpus_per_task
        self.constraint = self.config.constraint
        
    def CreateSlurmScript(self, cmd, slurm_filename):
        with open(slurm_filename, "w") as fh:
            fh.write("#!/bin/bash\n")
            fh.write("#SBATCH --nodes="+self.nodes+"\n")
            fh.write("#SBATCH --cpus-per-task="+self.cpus_per_task+"\n")
            fh.write("#SBATCH --constraint=\""+self.constraint+"\"\n")
            fh.write("echo \"Running on host: `hostname`\"\n")
            fh.write("echo \"SLURM_JOB_ID: $SLURM_JOB_ID\"\n")
            fh.write("echo \"SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST\"\n")
            fh.write("cd "+os.path.dirname(os.path.abspath(slurm_filename))+" ; \n")
            fh.write("echo \""+cmd+"\"\n")
            fh.write(cmd+"\n")
            fh.write("cd -\n")
            fh.write("echo \"Completed job!\"\n")
            fh.close()

    def GetSlurmJobCommand(self, slurm_filename):
        return "sbatch -W "+slurm_filename+" ; wait"

def GetObject(framework):
    obj = DeffeSlurm(framework)
    return obj
