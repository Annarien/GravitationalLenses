#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=<# tasks per node>
#SBATCH --mem=<memory in Mb, max is 64000>
#SBATCH --time=<time in HH:MM:SS format, max is 24:00:00>

source /home/mjh/SETUP_PY27.sh
module load python/2.7.8_gcc-4.4.7
python postageStampsDES.py
