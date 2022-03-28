#!/bin/sh 
########## Begin MOAB/Slurm header ##########
#
# Give job a reasonable name
#MOAB -N ENres2
#
# Request number of nodes and CPU cores per node for job
#MOAB -l nodes=2:ppn=20
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

echo 'Start now'
#source $( ws_find conda )/conda/etc/profile.d/conda.sh
ml devel/conda
conda activate pgnn
#cd $( ws_find conda )
cd ./physics_guided_nn
echo 'begin python'

python ./code/ENres2.py -d sparse

conda deactivate


