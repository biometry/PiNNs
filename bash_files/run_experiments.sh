#!/bin/sh 
########## Begin MOAB/Slurm header ##########
#
# Give job a reasonable name
#MOAB -N ensemble experiments run
#
# Request number of nodes and CPU cores per node for job
#MOAB -l nodes=2:ppn=20
#
# Estimated wallclock time for job
#MOAB -l walltime=00:02:00:00
#
# Write standard output and errors in same file
#MOAB -j oe 
#
# Send mail when job begins, aborts and ends
#MOAB -m bae
#
########### End MOAB header ##########

echo "Working Directory:                    $PWD"
echo "Running on host                       $HOSTNAME"
echo "Job id:                               $MOAB_JOBID"
echo "Job name:                             $MOAB_JOBNAME"
echo "Number of nodes allocated to job:     $MOAB_NODECOUNT"
echo "Number of cores allocated to job:     $MOAB_PROCCOUNT"


# Load conda
ml devel/conda
conda activate pgnn

cd physics_guided_nn/code

# Run experiment
python3 ENmlp.py
python3 -c 'import simulations; simulations.gen_simulations(n=10)'
python3 pretraining.py
python3 -c 'import evalmlpDA; evalmlpDA(da=2)'
