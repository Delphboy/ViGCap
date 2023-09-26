# ViGCap
A VisionGNN based image captioning architecture

## Model Overview
TODO: Architecture diagram here

## Setup
### Dependencies

The `setup.sh` script will configure the environment for the QMUL HPC facilities. This will provide a good starting point for setting up the repository on your local environment. 

```bash
# Andrena
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# EECS
python3 -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
```


```bash
python3 -m pip install pycocotools
python3 -m pip install spacy
python3 -m pip install tqdm
python3 -m pip install nltk
python3 -m pip install mosestokenizer
python3 -m pip install revtok
python3 -m pip install h5py
python3 -m pip install timm
python3 -m pip install tensorboard

python3 -m pip uninstall urllib3
python3 -m pip install urllib3==1.26.6
python -m spacy download en
```

### Data

The model is trained, validated, and tested on the COCO Karpathy split. This can be downloaded with the script below

```bash
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
```

## Usage

### Training procedure
Run `python train_vig.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--m` | Number of memory vectors (default: 40) |
| `--head` | Number of heads (default: 8) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|
| `--seed` | Specifies the fixed seed (default: -1 (random seed each time)) |
| `--test_every` | Run evaluation on the test set every N epochs (default -1 (disable)) |
| `--vig_size` | The model size of the ViG-based encoder [tiny | small | base] (default: tiny) |
| `--vig_type` | The model type of the ViG-based encoder [default | pyramid] (default: "default") |

For example, to train our model with the parameters used in our experiments, use
```bash
python3 train_vig.py --exp_name "ViGCap" \
                    --dataset "coco" \
                    --dataset_img_path "/data/coco" \
                    --dataset_ann_path "/data/coco/dataset_coco.json" \
                    --m 40 \
                    --head 8 \
                    --workers 4 \
                    --batch_size 32 \
                    --max_epochs 20 \
                    --vig_type pyramid \
                    --vig_size base
```


## Code References
- [Meshed-Memory Transformer for Image Captioning](https://github.com/aimagelab/meshed-memory-transformer)
- [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)