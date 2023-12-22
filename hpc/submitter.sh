#!/bin/bash

declare Ks=(10) # 15)
declare sag_ratios=(0.75) # 0.5 0.25)
declare dropouts=(0.1)
declare features=("m25") # "m50" "m75" "m100")

dataset="coco"

for feature in "${features[@]}"
do
    for k in "${Ks[@]}"
    do
        for sag_ratio in "${sag_ratios[@]}"
        do
            for dropout in "${dropouts[@]}"
            do
                echo "Submitting job with feats=${feature}, k=${k}, sag_ratio=${sag_ratio}, dropout=${dropout}"
                
                # Andrena
                qsub -v k=$k,feature=$feature,sag_ratio=$sag_ratio,dropout=$dropout -N "${feature}-gat-${dataset}-k_${k}-sag_${sag_ratio}-dropout_${dropout}" train.qsub 
            
                # JADE2
                # sbatch --export=k=$k,feature=$feature,sag_ratio=$sag_ratio,dropout=$dropout --job-name=${feature}-gat-${dataset}-k_${k}-sag_${sag_ratio}-dropout_${dropout} --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap/logs/${feature}-gat-${dataset}-k_${k}-sag_${sag_ratio}-dropout_${dropout}.out train.sh
            done
        done
    done
done


sleep 5

# Andrena
qstat

# JADE2
# squeue -u $(whoami)

echo "All jobs submitted"