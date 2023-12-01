#!/bin/bash

# declare learning_rates=(0.000164 0.00001 0.0005 0.0003)
# declare annealing_factors=(0.1 0.5 0.9)
# declare emb_sizes=(512 1024 2048)

declare learning_rates=(0.000164)
declare annealing_factors=(0.1)
declare emb_sizes=(512)

declare Ks=(5) # 10 15 20)
declare sag_ratios=(0.75)
declare dropouts=(0.1)

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
                        echo "Submitting job with lr=${lr}, af=${af}, k=${k}, meshed_emb_size=${emb_size}, patch_feature_size=${emb_size}, gnn_emb_size=${emb_size}, sag_ratio=${sag_ratio}, dropout=${dropout}"
                        
                        # Andrena
                        # qsub -v lr=$lr,af=$af,k=$k,meshed_emb_size=$emb_size,patch_feature_size=$emb_size,gnn_emb_size=$emb_size,sag_ratio=$sag_ratio,dropout=$dropout -N "coco-lr_${lr}-af_${af}-k_${k}-meshed_${emb_size}-patch_${emb_size}-gnn_${emb_size}-sag_${sag_ratio}-dropout_${dropout}" train.qsub 
                    
                        # JADE2
                        sbatch --export=lr=$lr,af=$af,k=$k,meshed_emb_size=$emb_size,patch_feature_size=$emb_size,gnn_emb_size=$emb_size,sag_ratio=$sag_ratio,dropout=$dropout --job-name=coco-lr_${lr}-af_${af}-k_${k}-meshed_${emb_size}-patch_${emb_size}-gnn_${emb_size}-sag_${sag_ratio}-dropout_${dropout} --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap/logs/coco-lr_${lr}-af_${af}-k_${k}-meshed_${emb_size}-patch_${emb_size}-gnn_${emb_size}-sag_${sag_ratio}-dropout_${dropout}.out train.sh
                    done
                done
            done
        done
    done
done

sleep 5

# Andrena
# qstat

# JADE2
squeue -u $(whoami)

echo "All jobs submitted"