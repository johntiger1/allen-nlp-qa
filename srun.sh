#!/bin/bash
source activate hurtfulwords
# srun script for running
srun  --mem=15g -c 2 -p gpu --gres=gpu:1 --unbuffered  python QADir/main.py
