#!/bin/bash

cd ./physics_guided_nn

input=("mlp" "res" "res2" "reg" "emb")

for i in ${input[*]}
do
    echo "submitting ${i}ft.sh"
    msub bash_files/${i}ft.sh
done
