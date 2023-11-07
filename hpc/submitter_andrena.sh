#!/bin/bash

declare learning_rates=(0.0001)
declare annealing_factors=(0.1) # 0.5 0.9)

declare Ks=(3 5 7) # 10 15 20)
declare emb_sizes=(512) # 1024)
declare sag_ratios=(0.1 0.25 0.5 0.75)
declare dropouts=(0.1 0.25 0.5)

for lr in "${learning_rates[@]}"
do
    for af in "${annealing_factors[@]}"
    do
        for k in "${Ks[@]}"
        do
            for emb_size in "${emb_sizes[@]}"
            do
                for sag_ratio in "${sag_ratios[@]}"
                do
                    for dropout in "${dropouts[@]}"
                    do
                        echo "Submitting job with lr=${lr}, af=${af}, k=${k}, meshed_emb_size=${meshed_emb_size}, patch_feature_size=${patch_feature_size}, gnn_emb_size=${gnn_emb_size}, sag_ratio=${sag_ratio}, dropout=${dropout}"
                        qsub -v lr=$lr,af=$af,k=$k,meshed_emb_size=$emb_size,patch_feature_size=$emb_size,gnn_emb_size=$emb_size,sag_ratio=$sag_ratio,dropout=$dropout -N "coco-lr_${lr}-af_${af}-k_${k}-meshed_${emb_size}-patch_${emb_size}-gnn_${emb_size}-sag_${sag_ratio}-dropout_${dropout}" train.qsub 
                    done
                done
            done
        done
    done
done

sleep 5
qstat

echo "All jobs submitted"