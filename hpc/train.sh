#!/bin/bash

#SBATCH --chdir=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=small

module load cuda/10.2
module load python/3.8.6

source .venv/bin/activate

# convert nBlocks to int
nBlocks=$(echo $nBlock | bc)
echo $nBlocks
python train_vig.py --dataset "coco" \
                    --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco" \
                    --dataset_ann_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco/dataset_coco.json" \
                    --exp_name "noGCN-COCO-ViGCap-${vigType}-${vigSize}" \
                    --batch_size 8 \
                    --m 40 \
                    --head 8 \
                    --workers 4 \
                    --batch_size 64 \
                    --max_epochs 20 \
                    --vig_type ${vigType} \
                    --vig_size ${vigSize} \
                    --patience -1 \
                    --dropout 0.0 \
                    --n_blocks ${nBlocks} \
                    --seed 42


# python train_vig.py --dataset "flickr8k" \
#                     --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/flickr8k/images" \
#                     --dataset_ann_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/flickr8k/dataset_flickr8k.json" \
#                     --exp_name "noGCN-flickr8k-ViGCap-${vigType}-${vigSize}" \
#                     --batch_size 8 \
#                     --m 40 \
#                     --head 8 \
#                     --workers 4 \
#                     --batch_size 64 \
#                     --max_epochs 20 \
#                     --vig_type ${vigType} \
#                     --vig_size ${vigSize} \
#                     --patience -1 \
#                     --dropout 0.5 \
#                     --n_blocks ${nBlocks} \
#                     --seed 42