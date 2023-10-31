#!/bin/bash

declare learning_rates=(0.0001 0.002)
declare annealing_factors=(0.1 0.5 0.9)

for lr in "${learning_rates[@]}"
do
    for af in "${annealing_factors[@]}"
    do
        qsub -v lr=$lr,af=$af -N "coco_lr${lr}_af${af}" train.qsub 
    done
done

sleep 5
qstat

echo "All jobs submitted"