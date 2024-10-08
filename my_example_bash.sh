#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16Gb
#SBATCH --job-name=example_run
#SBATCH --time=00:10:00
#SBATCH --output=example_run.%j

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
#echo "Job id:                               $SBATCH_JOBID"
#echo "Job name:                             $SBATCH_JOBNAME"
#echo "Number of nodes allocated to job:     $SBATCH_NODECOUNT"
#echo "Number of cores allocated to job:     $SBATCH_PROCCOUNT"

# generic settings
WORKDIR=/home/${USER}/PycharmProjects/PiNNs

cd $WORKDIR
module purge
module load conda
conda activate /your/conda/envs/pinns

srun python3 temporal/nas/ENres.py -d full

