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
lr_cleaned=$(echo "$lr" | bc)
k_cleaned=$(echo "$k" | bc)
meshed_emb_size_cleaned=$(echo "$meshed_emb_size" | bc)
gnn_emb_size_cleaned=$(echo "$gnn_emb_size" | bc)
sag_ratio_cleaned=$(echo "$sag_ratio" | bc)
dropout_cleaned=$(echo "$dropout" | bc)

                    # --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco" \
python train_vig.py --dataset ${dataset} \
                    --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/m50/" \
                    --dataset_ann_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco/dataset_coco.json" \
                    --feature_limit 50 \
                    --exp_name "m50-gat-${dataset}-lr_${lr}-k_${k}-meshed_${meshed_emb_size_cleaned}-gnn_${gnn_emb_size_cleaned}-sag_${sag_ratio}-dropout_${dropout}" \
                    --m 40 \
                    --workers 4 \
                    --max_epochs 5 \
                    --batch_size 32 \
                    --seed 42 \
                    --patience 5 \
                    --force_rl_after -1 \
                    --learning_rate $lr_cleaned \
                    --k $k_cleaned \
                    --meshed_emb_size $meshed_emb_size_cleaned \
                    --patch_feature_size 2048 \
                    --gnn_emb_size $gnn_emb_size_cleaned \
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
