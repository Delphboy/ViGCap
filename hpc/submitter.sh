#!/bin/bash

# declare learning_rates=(0.000164)

declare learning_rates=(1)

declare gnn_emb_sizes=(2048)
declare m2_emb_sizes=(2048)

declare Ks=(5)
declare sag_ratios=(0.75) # 0.5 0.25)
declare dropouts=(0.25)

dataset="coco"

for lr in "${learning_rates[@]}"
do
    for k in "${Ks[@]}"
    do
        for gnn_emb_size in "${gnn_emb_sizes[@]}"
        do
            for m2_emb_size in "${m2_emb_sizes[@]}"
            do
                for sag_ratio in "${sag_ratios[@]}"
                do
                    for dropout in "${dropouts[@]}"
                    do
                        echo "Submitting job with lr=${lr}, k=${k}, meshed_emb_size=${m2_emb_size}, patch_feature_size=2048, gnn_emb_size=${gnn_emb_size}, sag_ratio=${sag_ratio}, dropout=${dropout}"
                        
                        # Andrena
                        # qsub -v lr=$lr,af=$af,k=$k,meshed_emb_size=$emb_size,patch_feature_size=$emb_size,gnn_emb_size=$emb_size,sag_ratio=$sag_ratio,dropout=$dropout -N "${dataset}-lr_${lr}-af_${af}-k_${k}-meshed_2048-patch_${emb_size}-gnn_${emb_size}-sag_${sag_ratio}-dropout_${dropout}" train.qsub 
                    
                        # JADE2
                        sbatch --export=lr=$lr,k=$k,meshed_emb_size=$m2_emb_size,gnn_emb_size=$gnn_emb_size,sag_ratio=$sag_ratio,dropout=$dropout --job-name=m50-gat-${dataset}-lr_${lr}-k_${k}-meshed_${m2_emb_size}-gnn_${gnn_emb_size}-sag_${sag_ratio}-dropout_${dropout} --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap/logs/m50-gat-${dataset}-lr_${lr}-k_${k}-meshed_${m2_emb_size}-gnn_${gnn_emb_size}-sag_${sag_ratio}-dropout_${dropout}.out train.sh
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