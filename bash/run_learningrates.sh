#!/bin/sh
cd ./physics_guided_nn/bash

for d in full sparse
do
    for i in mlp res res2 reg
    do
	FILE="${i}ft_${d}.sh"
	if [ -f "$FILE" ];then
	    echo "$FILE exists"
	    rm $FILE
	    touch $FILE
	else 
            echo "$FILE does not exist"
	    touch $FILE
	fi

	echo -e "#!/bin/sh\n#MOAB -N ft_EX1_${i}_${d}\n#MOAB -l nodes=1:ppn=20\n#MOAB -l walltime=02:00:00:00\n#MOAB -j oe \n" >> $FILE
	echo -e "ml devel/conda\nconda activate pgnn\ncd ./physics_guided_nn\npython code/${i}ft.py -d ${d}\nconda deactivate" >> $FILE
	msub $FILE
    done
done    

