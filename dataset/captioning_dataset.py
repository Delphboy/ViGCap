import json
import os
from collections import Counter
from itertools import chain
from typing import List

import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def preprocess_caption(caption: str) -> str:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    caption = caption.lower()

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]

    for char in non_alphanumeric:
        caption = caption.replace(char, "")
    while "  " in caption:
        caption = caption.replace("  ", " ")

    return caption.strip()


def preprocess_captions(captions: List[str]) -> List[str]:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    captions = [caption.lower() for caption in captions]

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]
    cleaned = []

    for sentence in captions:
        for char in non_alphanumeric:
            sentence = sentence.replace(char, "")
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
        cleaned.append(sentence.strip())
    return cleaned


class Vocab:
    def __init__(self, freq_threshold, dataset_name="coco"):
        self.itos = {1: "<pad>", 2: "<bos>", 3: "<eos>", 0: "<unk>"}
        self.stoi = {"<pad>": 1, "<bos>": 2, "<eos>": 3, "<unk>": 0}
        self.freq_threshold = freq_threshold
        self.talk_file_location = f"data/{dataset_name}_talk.json"

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        if os.path.exists(self.talk_file_location):
            with open(self.talk_file_location, "r") as f:
                self.itos = json.load(f)
                self.stoi = {v: int(k) for k, v in self.itos.items()}
            return

        frequencies = Counter(
            word for sentence in sentence_list for word in sentence.split(" ")
        )
        idx = 4  # idx 1, 2, 3, 0 are already taken (PAD, SOS, EOS, UNK)
        for word, count in frequencies.items():
            if count == self.freq_threshold:
                self.stoi[word] = int(idx)
                self.itos[idx] = word
                idx += 1

        # write self.itos to a json file
        with open(self.talk_file_location, "w") as f:
            json.dump(self.itos, f)

    def numericalize(self, text):
        # text is a string
        # we want to return a list of integers
        # providing a numericalized version of the text
        tokenized_text = text.split(" ")
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]


# Adapted to give BU-TD or Superpixel features
class CaptioningDatasetWithFeatures(Dataset):
    def __init__(
        self,
        root_dir: str,
        captions_file: str,
        dataset_name: str = "coco",
        freq_threshold: int = 5,
        split: str = "train",
        feature_limit: int = 50,
    ):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.freq_threshold = freq_threshold
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split
        self.feature_limit = feature_limit

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.captions = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                img_path = os.path.join(
                    self.root_dir,
                    f"{image_data['cocoid']}.npz"
                    if dataset_name == "coco"
                    else image_data["filename"].split(".")[0] + ".npz",
                )

                caps = [
                    " ".join(sentence["tokens"]) for sentence in image_data["sentences"]
                ]
                self.image_locations.append(img_path)
                self.captions.append(caps)

        self.length = len(self.captions)
        self.vocab = Vocab(freq_threshold, dataset_name)
        self.vocab.build_vocabulary(self.text)

    def __getitem__(self, index):
        img_path = self.image_locations[index]
        image = np.load(img_path)["feat"]
        image = torch.from_numpy(image)

        # if image.shape[0] < self.feature_limit then we need to pad it with zeros
        if image.shape[0] < self.feature_limit:
            pad = torch.zeros((self.feature_limit - image.shape[0], 2048))
            image = torch.cat((image, pad), 0)

        captions = self.captions[index]

        # randomly select a caption from the list of captions
        caption = "<bos> " + np.random.choice(captions) + " <eos>"
        seq = self.vocab.numericalize(caption)

        return image[:self.feature_limit], seq, captions

    def __len__(self):
        return self.length

    @property
    def text(self):
        return list(chain.from_iterable(caption_list for caption_list in self.captions))

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[str(wi.item())]
                if word == "<eos>":
                    break
                if word == "<bos>":
                    continue
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions


# Orginal - Gives images and captions
class CaptioningDatasetWithImages(Dataset):
    def __init__(
        self,
        root_dir: str,
        captions_file: str,
        dataset_name: str = "coco",
        freq_threshold: int = 5,
        split: str = "train",
    ):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.freq_threshold = freq_threshold
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.captions = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                img_path = os.path.join(
                    self.root_dir,
                    image_data.get("filepath", ""),
                    image_data["filename"],
                )

                caps = [
                    " ".join(sentence["tokens"]) for sentence in image_data["sentences"]
                ]
                self.image_locations.append(img_path)
                self.captions.append(caps)

        self.length = len(self.captions)
        self.vocab = Vocab(freq_threshold, dataset_name)
        self.vocab.build_vocabulary(self.text)

        # numericalise the first caption of each image and store it in self.seq
        # add in the <bos> and <eos> tokens
        self.seq = []
        for caption_list in self.captions:
            caption = "<bos> " + caption_list[0] + " <eos>"
            caption = self.vocab.numericalize(caption)
            self.seq.append(caption)

    def __getitem__(self, index):
        img_path = self.image_locations[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        seq = self.seq[index]
        captions = self.captions[index]

        return image, seq, captions

    def __len__(self):
        return self.length

    @property
    def text(self):
        return list(chain.from_iterable(caption_list for caption_list in self.captions))

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[str(wi.item())]
                if word == "<eos>":
                    break
                if word == "<bos>":
                    continue
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions


class Batcher:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        # def sort(batch):
        #     batch.sort(key=lambda x: len(x[1]), reverse=True)
        #     return batch

        # batch = sort(batch)
        images, seq, captions = zip(*batch)

        images = torch.stack(images, 0)

        max_len = max(len(s) for s in seq)

        # Pad the sequences so that they are all max length
        # sequences is a list of ints
        # we want to just add <pad> tokens to the end of the sequence so that they
        # are all max_len in length
        seq_padded = []
        for s in seq:
            padded = s + [self.vocab.stoi["<pad>"]] * (max_len - len(s))
            seq_padded.append(padded)

        seq_padded = torch.LongTensor(seq_padded)

        return images, seq_padded, captions
