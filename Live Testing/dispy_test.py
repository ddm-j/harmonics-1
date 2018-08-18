#!/software/development/Build/Anaconda3-4.4.0/envs/python-3.6/bin/python -u
#SBATCH --output=dr.txt

#

#SBATCH --ntasks=5
#SBATCH --time=10:00
#SBATCH --nodes=5
#SBATCH --mem-per-cpu=10

# Import System Related Packages & Append Path
import os
import subprocess
import sys
sys.path.append(os.getcwd())


# Import Other Packages
import dispy
import numpy as np
import random
import time
import socket


nodes = subprocess.check_output('echo $SLURM_JOB_NODELIST',shell=True)
print(nodes)
nodes = nodes.decode('utf-8')
nodes = nodes[8:-2]
nodes = [int(nodes[0]),int(nodes[-1])]

print(nodes)

#for i in nodes:

	#os.system('srun dispynode.py --clean --daemon & &>/dev/null')

os.system('srun dispynode.py --clean --daemon &')

def compute(n):
	host = socket.gethostname()
	return (host,n**2)

x = np.arange(1000)
x = [random.randint(1,10) for i in x]

# Set up parallel computing task

cluster = dispy.JobCluster(compute)
cluster.print_status()

print('Cluster Created')
jobs = []
for i in x:
	job = cluster.submit(i)
	job.id = i
	jobs.append(job)

print('Calculating Jobs')
for job in jobs:
	host, n = job()
	#print('%s computed square of %s at %s = %s' % (host,job.id,job.start_time,n))


cluster.print_status()
