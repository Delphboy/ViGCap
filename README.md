# ViGCap
A VisionGNN based image captioning architecture

## Model Overview
TODO: Architecture diagram here

## Setup
### Dependencies

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

Annotations can be downloaded using the following:

```bash
cd data
wget https://ailb-web.ing.unimore.it/publicfiles/drive/meshed-memory-transformer/annotations.zip
unzip annotations.zip
rm annotations.zip
```

Detections can be downloaded using the following (Note: this file is large! (~53.5 GB)):

```bash
cd data
wget https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/coco_detections.hdf5
```

## Usage

### Evaluation
To reproduce the results reported in our paper, download the pretrained model file [meshed_memory_transformer.pth](https://ailb-web.ing.unimore.it/publicfiles/drive/meshed-memory-transformer/meshed_memory_transformer.pth) and place it in the code folder.

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

#### Expected output
Under `output_logs/`, you may also find the expected output of the evaluation code.


### Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--m` | Number of memory vectors (default: 40) |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

For example, to train our model with the parameters used in our experiments, use
```
python train.py --exp_name m2_transformer --batch_size 50 --m 40 --head 8 --warmup 10000 --features_path /path/to/features --annotation_folder /path/to/annotations
```


## Code References
- [Meshed-Memory Transformer for Image Captioning](https://github.com/aimagelab/meshed-memory-transformer)
- [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)