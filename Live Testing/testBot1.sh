#!/bin/bash

#

#SBATCH --job-name=test

#SBATCH --output=res.txt

#

#SBATCH --ntasks=1

#SBATCH --time=10:00

#SBATCH --mem-per-cpu=100

module load python-dev/3.6

python botProto1.py -frame 1year -pairs EUR_USD GBP_USD AUD_USD NZD_USD -parameters 0.5 15 15 21 --risk 1




