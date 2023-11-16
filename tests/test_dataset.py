import pytest
import torch
from torch.utils.data import DataLoader

from dataset.captioning_dataset import (
    Batcher,
    CaptioningDataset,
    Vocab,
    preprocess_caption,
    preprocess_captions,
)


@pytest.mark.parametrize(
    "expr, expected_result",
    [
        ("This is a bAD caption", "this is a bad caption"),
        ("This is another bad caption", "this is another bad caption"),
        ("  Yet another bad caption  !!!", "yet another bad caption"),
    ],
)
def test_given_single_unformatted_caption_when_preprocessing_then_caption_is_formatted(
    expr, expected_result
):
    # Arrange
    bad_caption = "This is a  ?   bAD caption!!!"
    good_caption = "this is a bad caption"

    # Act
    formatted_caption = preprocess_caption(expr)

    # Assert
    assert formatted_caption == expected_result


def test_given_multiple_unformatted_captions_when_preprocessing_then_captions_are_formatted_correctly():
    # Arrange
    bad_captions = [
        "This is a bAD caption",
        "This is another bad caption",
        "  Yet another bad caption  !!!",
    ]
    good_captions = [
        "this is a bad caption",
        "this is another bad caption",
        "yet another bad caption",
    ]

    # Act
    formatted_captions = preprocess_captions(bad_captions)

    # Assert
    assert formatted_captions == good_captions


def test_given_caption_string_when_numericalising_then_encoded_correctly():
    # Arrange
    caption = "a dog and a cat"
    vocab = Vocab(freq_threshold=1, dataset_name="coco")
    vocab.build_vocabulary([caption])
    expected_result = [4, 191, 17, 4, 112]

    # Act
    encoded_caption = vocab.numericalize(caption)

    # Assert
    assert encoded_caption == expected_result


def test_given_captioning_dataset_when_getting_data_then_captions_match_seq():
    # Arrange
    val_data = CaptioningDataset(
        "/import/gameai-01/eey362/datasets/flickr8k/images",
        "/homes/hps01/ViGCap/data/karpathy_splits/dataset_flickr8k.json",
        dataset_name="flickr8k",
        freq_threshold=5,
        split="val",
    )

    # Act
    img, seq, cap = val_data.__getitem__(0)
    test_seq = [
        2,
        7,
        80,
        97,
        353,
        32,
        5,
        4,
        92,
        11,
        215,
        2693,
        222,
        7,
        211,
        50,
        193,
        80,
        3,
    ]

    # Assert
    assert seq == test_seq


def test_given_captioning_dataset_when_decoding_seq_then_correct_caption_given():
    # Arrange
    val_data = CaptioningDataset(
        "/import/gameai-01/eey362/datasets/flickr8k/images",
        "/homes/hps01/ViGCap/data/karpathy_splits/dataset_flickr8k.json",
        dataset_name="flickr8k",
        freq_threshold=5,
        split="val",
    )
    img, seq, caps = val_data.__getitem__(0)

    # Act
    decoded_caption = val_data.decode(torch.tensor([seq]))[0]

    # Assert
    assert decoded_caption == caps[0]


def test_given_dataloader_when_calling_batch_then_data_is_correct():
    # Arrange
    val_data = CaptioningDataset(
        "/import/gameai-01/eey362/datasets/flickr8k/images",
        "/homes/hps01/ViGCap/data/karpathy_splits/dataset_flickr8k.json",
        dataset_name="flickr8k",
        freq_threshold=5,
        split="val",
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=Batcher(val_data.vocab),
    )

    # Act
    images, seq, caps = next(iter(val_dataloader))
    gen_seq = [val_data.vocab.numericalize(cap[0]) for cap in caps]

    # Assert
    assert any(
        [
            all([gen_seq[i][j] in seq[i] for j in range(len(gen_seq[i]))])
            for i in range(len(gen_seq))
        ]
    )
