#!/bin/bash

declare -a vigTypes=("default")
declare -a vigSizes=("base")
declare -a nBlocks=("16")

for vigType in "${vigTypes[@]}"
do
    for vigSize in "${vigSizes[@]}"
    do
        for nBlock in "${nBlocks[@]}"
        do
            echo "Submitting job for vigType: $vigType and vigSize: $vigSize"
            sbatch --export=vigType=$vigType,vigSize=$vigSize,nBlock=$nBlock --job-name=noGCN_coco_vigcap_${vigType}_${vigSize} --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap/logs/noGCN_coco_ViGCap-${vigType}-${vigSize}.out train.sh
        done
    done
done

echo "All jobs submitted"
squeue -u $(whoami)