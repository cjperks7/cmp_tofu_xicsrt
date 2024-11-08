#!/bin/bash
#SBATCH --job-name=XICSRT
#SBATCH -C rocky8
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=500G
#SBATCH -p sched_mit_psfc_r8
#SBATCH --output=output/XRSHRKr_res_v1/slurm.out
#SBATCH --error=output/XRSHRKr_res_v1/slurm.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cjperks@mit.edu

# Init
module purge
. $HOME/bin/init_R8_py310.sh

# Your job script here
python resolution_HPC.py XRSHRKr_res_v1
