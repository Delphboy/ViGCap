#!/bin/bash
#SBATCH --job-name=train
#SBATCH --chdir=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap
#SBATCH --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/ViGCap/logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=devel

module load cuda/10.2
module load python/3.8.6

python train_vig.py --dataset "coco" \
                    --dataset_img_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco" \
                    --dataset_ann_path "/jmain02/home/J2AD007/txk47/hxs67-txk47/coco/dataset_coco.json" \
                    --exp_name debug-coco \
                    --batch_size 8 \
                    --m 40 \
                    --head 8 \
                    --workers 4