#!/bin/bash

USERNAME=eey362
SCRATCH_DIR=/data/scratch/$USERNAME/
DATASET_DIR=$SCRATCH_DIR/datasets/coco

ml load python

# Setup repo
echo "Setting up repo"

cd ~
git clone git@github.com:Delphboy/ViGCap.git
cd ViGCap
mkdir logs
mkdir saved_models
mkdir tensorboard_logs

echo "Setting up python environment"
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -r requirements.txt

# python3 -m pip install pycocotools
# python3 -m pip install spacy
# python3 -m pip install tqdm
# python3 -m pip install nltk
# python3 -m pip install mosestokenizer
# python3 -m pip install revtok
# python3 -m pip install h5py
# python3 -m pip install timm
# python3 -m pip install tensorboard

# python3 -m pip uninstall -y urllib3
# python3 -m pip install urllib3==1.26.6
# python -m spacy download en

# python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Use sed to replace the string "eey362" with your $USERNAME in hpc/train.qsub
sed -i "s/eey362/$USERNAME/g" hpc/train.qsub


# Download coco to Andrena scratch storage
echo "Downloading coco to $DATASET_DIR"
cd $SCRATCH_DIR

mkdir datasets
cd datasets
mkdir coco
cd coco

wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

unzip train2014.zip
unzip val2014.zip
unzip test2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip

wget https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download= -O dataset_coco.json


echo "Good to go!"
