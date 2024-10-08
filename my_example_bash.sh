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
conda activate /perm/pamw/conda/envs/pinns

srun python3 temporal/nas/ENmlp.py -d full

#srun python3 src/forecast/run_forecast_global.py mlp_europe_config.yaml lstm_config_sc.yaml xgb_europe_config.yaml # lstm_config.yaml xgb_europe_config.yaml
#srun python3 src/forecast/run_forecast_global.py --config_file_mlp mlp_europe_config.yaml --config_file_lstm lstm_europe_config.yaml --config_file_xgb xgb_europe_config.yaml # mlp_global_config.yaml lstm_config_sc.yaml xgb_global_config.yaml 
#srun python3 src/forecast/run_forecast.py --config_file_mlp None --config_file_lstm lstm_global_config.yaml --config_file_xgb None #xgb_global_config.yaml
