#!/bin/bash

#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iliad1
#SBATCH --output=/iliad/u/jhejna/slurm_logs/baseline_%A.out
#SBATCH --error=/iliad/u/jhejna/slurm_logs/baseline_%A.err
#SBATCH --job-name="vit_baseline"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

. /iliad2/u/jhejna/data4robotics/setup_shell.sh
cd /iliad2/u/jhejna/data4robotics

python finetune.py agent/features=vit_base agent.features.restore_path=/iliad/u/jhejna/data4robotics/vc1.pth task=coffee_pod exp_name=coffee_pod