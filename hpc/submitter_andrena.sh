#!/bin/bash


declare -a vigTypes=("default") # "pyramid")
declare -a vigSizes=("tiny") # "small" "base")
declare -a kValues=("9" "11" "13" "21")
declare -a gnnTypes=("mr" "edge" "sage" "gin")

for vigType in "${vigTypes[@]}"
do
    for vigSize in "${vigSizes[@]}"
    do
        for k in "${kValues[@]}"
        do
            for gnnType in "${gnnTypes[@]}"
            do
                echo "Submitting job for vigType: $vigType and vigSize: $vigSize with k: $k and gnnType: $gnnType"
                qsub -v vigType=$vigType,vigSize=$vigSize,k=$k,gnnType=$gnnType -N "Vigcap_${vigType}_${vigSize}_${k}_${gnnType}" train.qsub
                done
        done
    done
done

echo "All jobs submitted"