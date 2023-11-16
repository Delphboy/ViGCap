from typing import Optional, Tuple

from torch.utils.data import DataLoader

from dataset.captioning_dataset import Batcher, CaptioningDataset, Vocab
from models.meshed.captioning_model import CaptioningModel
from models.meshed.transformer import (
    MemoryAugmentedEncoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
)
from models.vig_cap import VigCap


def get_training_data(
    args,
) -> Tuple[CaptioningDataset, CaptioningDataset, CaptioningDataset]:
    train_data = CaptioningDataset(
        args.dataset_img_path,
        args.dataset_ann_path,
        dataset_name=args.dataset,
        freq_threshold=5,
        split="train",
    )
    val_data = CaptioningDataset(
        args.dataset_img_path,
        args.dataset_ann_path,
        dataset_name=args.dataset,
        freq_threshold=5,
        split="val",
    )
    test_data = CaptioningDataset(
        args.dataset_img_path,
        args.dataset_ann_path,
        dataset_name=args.dataset,
        freq_threshold=5,
        split="test",
    )
    return train_data, val_data, test_data


def get_dataloader(
    dataset: CaptioningDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    num_workers = 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=Batcher(dataset.vocab),
    )


def get_model(args: any, vocab: Vocab) -> CaptioningModel:
    encoder = MemoryAugmentedEncoder(
        args.n,
        vocab.stoi["<pad>"],
        attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={"m": args.m},
        d_in=args.meshed_emb_size,
        dropout=args.dropout,
    )
    decoder = MeshedDecoder(
        len(vocab), 54, args.n, vocab.stoi["<pad>"], dropout=args.dropout
    )
    model = VigCap(vocab.stoi["<bos>"], encoder, decoder, args)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model
