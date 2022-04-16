#!/bin/sh
cd physics_guided_nn/bash

for d in full sparse
do
    for i in mlp res res2 reg
    do
	FILE="EN${i}_${d}.sh"
	if [ -f "$FILE" ];then
	    echo "$FILE exists"
	    rm $FILE
	    touch $FILE
	else 
            echo "$FILE does not exist"
	    touch $FILE
	fi

	echo -e "#!/bin/sh\n#SBATCH --job-name NAS_EX1_${i}_${d}\n#SBATCH --account=Project_2000527\n#SBATCH --time=10-00:15:00\n#SBATCH --mem-per-cpu=100G\n#SBATCH --partition=longrun\n" >> $FILE
	echo -e "source /projappl/project_2000527/miniconda3/etc/profile.d/conda.sh\nconda activate pgnn\ncd /users/mosernik/physics_guided_nn\npython code/EN${i}.py -d ${d}\nconda deactivate" >> $FILE
	sbatch $FILE
    done
done    

