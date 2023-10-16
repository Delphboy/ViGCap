from typing import Optional, Tuple

from torch.utils.data import DataLoader

from data.captioning_dataset import CaptioningDataset, CocoBatcher, Vocab
from models.new_vig_cap import VigCap
from models.transformer import (
    MemoryAugmentedEncoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
)


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
    dataset_name: Optional[str] = "coco",
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    talk_file_location = f"data/{dataset_name}_talk.json"
    num_workers = 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=CocoBatcher(talk_file_location),
    )


def get_model(args: any, vocab: Vocab) -> VigCap:
    emb_size = 512  # TODO: make this a parameter with args
    encoder = MemoryAugmentedEncoder(
        3,
        0,
        d_in=emb_size,
        attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={"m": 40},
    )
    decoder = MeshedDecoder(len(vocab), 54, 3, vocab.stoi["<PAD>"])
    model = VigCap(vocab.stoi["<SOS>"], encoder, decoder, emb_size)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model
