#!/bin/bash


declare -a vigTypes=("default" "pyramid")
declare -a vigSizes=("tiny" "small" "base")

for vigType in "${vigTypes[@]}"
do
    for vigSize in "${vigSizes[@]}"
    do
        echo "Submitting job for vigType: $vigType and vigSize: $vigSize"
        qsub -v vigType=$vigType,vigSize=$vigSize -N "vigcap_${vigType}_${vigSize}" train.qsub
    done
done

echo "All jobs submitted"