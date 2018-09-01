#!/bin/bash

#sbatch dispy_bot.py -frame 2year -pairs EUR_USD GBP_USD NZD_USD AUD_USD -parameters 5 5 5 --risk 1

sbatch window_bot.py

#sbatch example_test.py
