#!/bin/bash

#SBATCH --chdir=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=small

module load cuda/10.2
module load python/3.8.6

source .venv/bin/activate

dataset="coco"

# convert nBlocks to int
k_cleaned=$(echo "$k" | bc)
sag_ratio_cleaned=$(echo "$sag_ratio" | bc)
dropout_cleaned=$(echo "$dropout" | bc)

if [ "$feature" == "m25" ]; then
    feature_limit=16
elif [ "$feature" == "m50" ]; then
    feature_limit=36
elif [ "$feature" == "m75" ]; then
    feature_limit=55
elif [ "$feature" == "m100" ]; then
    feature_limit=77
else
    echo "Invalid feature"
    exit 1
fi


python train_vig.py --dataset ${dataset} \
                    --dataset_ann_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco/dataset_coco.json" \
                    --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/${feature}/" \
                    --feature_limit ${feature_limit} \
                    --exp_name "${feature}-2gat-${dataset}-k_${k}-sag_${sag_ratio}-dropout_${dropout}" \
                    --m 40 \
                    --workers 2 \
                    --max_epochs 30 \
                    --batch_size 32 \
                    --seed 42 \
                    --patience 5 \
                    --force_rl_after -1 \
                    --k $k_cleaned \
                    --meshed_emb_size 512 \
                    --patch_feature_size 2048 \
                    --gnn_emb_size 2048 \
                    --sag_ratio $sag_ratio_cleaned \
                    --dropout $dropout_cleaned \

# python train_vig.py --dataset "flickr8k" \
#                     --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/flickr8k/images" \
#                     --dataset_ann_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/flickr8k/dataset_flickr8k.json" \
#                     --exp_name "Vigcap_flickr8k-${lr}-${af}-${meshed_emb_size_cleaned}" \
#                     --m 40 \
#                     --workers 4 \
#                     --max_epochs 18 \
#                     --batch_size 32 \
#                     --seed 42 \
#                     --learning_rate $lr_cleaned \
#                     --anneal $af_cleaned \
#                     --k $k_cleaned \
#                     --meshed_emb_size $meshed_emb_size_cleaned \
#                     --patch_feature_size $patch_feature_size_cleaned \
#                     --gnn_emb_size $gnn_emb_size_cleaned \
#                     --sag_ratio $sag_ratio_cleaned \
#                     --dropout $dropout_cleaned \
#                     --patience -1
