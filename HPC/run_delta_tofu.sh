#!/bin/bash
#SBATCH --job-name=XICSRT
#SBATCH -C rocky8
#SBATCH --array=0
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=500G
#SBATCH -p sched_mit_psfc_r8
#SBATCH --output=output/mv_v1/slurm_%a.out
#SBATCH --error=output/mv_v1/slurm_%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cjperks@mit.edu

# Init
module purge
. $HOME/bin/init_R8_py310.sh

# Your job script here
python delta_HPC.py $SLURM_ARRAY_TASK_ID mv_v1 tofu
