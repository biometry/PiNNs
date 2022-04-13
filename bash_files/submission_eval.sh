#!/bin/bash

cd ./physics_guided_nn

input=("mlp" "res" "res2" "reg" )

for i in ${input[*]}
do
    echo "submitting eval${i}.sh"
    msub bash_files/eval${i}.sh
done
