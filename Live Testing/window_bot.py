#!/software/development/Build/Anaconda3-4.4.0/envs/python-3.6/bin/python -u
#SBATCH --output=dr.txt

#

#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G


# MAIN SCRIPT SECTION
# Scipt imports

import os
import sys
print('*** Python interpreter version ***\n')
print(sys.version,'\n')
sys.path.append(os.getcwd())
import subprocess
import warnings
import os.path
import optimization_tools
import time

# Initiate Dispy Node Server on Compute Nodes

print('*** Joker Computational Node Allocation ***')

nodes = subprocess.check_output('echo $SLURM_JOB_NODELIST',shell=True)
nodes = nodes.decode('utf-8')
nodes = nodes[8:-2]
#nodes = [int(nodes[0]),int(nodes[-1])]

print('Joker', nodes, '\n')
print('*** Dispy Server Startup Messages ***')
os.system('srun dispynode.py --clean --daemon &')
print('\n')


warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings('ignore',category=UserWarning)

t0 = time.time()

opt = optimization_tools.dispy_optimizer(frame='5year')
opt.prep()
results, equity = opt.search()
results.to_csv('window_results_scipy.csv')
equity.to_csv('equity_results_scipy.csv')

best_idx = results.sharpe.idxmax()
print(results.iloc[best_idx])

t1 = time.time()

print('Overall Backtest Took: ',t1-t0, 'seconds')