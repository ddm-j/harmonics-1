#!/bin/bash

sbatch dispy_bot.py -frame 5year -pairs EUR_USD GBP_USD NZD_USD AUD_USD -parameters 5 5 5 --risk 1

#sbatch dispy_opt.py -frame 1year
