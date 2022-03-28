#!/bin/sh 
########## Begin MOAB/Slurm header ##########
#
# Give job a reasonable name
#MOAB -N Simulations for Domain Adaptation
#
# Request number of nodes and CPU cores per node for job
#MOAB -l nodes=4:ppn=20
#
# Estimated wallclock time for job
#MOAB -l walltime=00:10:00:00
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

# Activate environment
cd physics_guided_nn/code
conda activate pgnn

# Run script with python3
python3  -c 'import simulations; simulations.gen_simulations(n=1000)'
