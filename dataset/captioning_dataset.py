import json
import os
from typing import List

import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def preprocess_captions(captions: List[str]) -> List[str]:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    captions = [caption.lower() for caption in captions]

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]
    cleaned = []

    for sentence in captions:
        for char in non_alphanumeric:
            sentence = sentence.replace(char, "")
        cleaned.append(sentence.strip())
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
    return cleaned


class Vocab:
    def __init__(self, freq_threshold, dataset_name="coco"):
        self.itos = {1: "<pad>", 2: "<bos>", 3: "<eos>", 0: "<unk>"}
        self.stoi = {"<pad>": 1, "<bos>": 2, "<eos>": 3, "<unk>": 0}
        self.freq_threshold = freq_threshold
        self.talk_file_location = f"data/{dataset_name}_talk.json"

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list=None):
        if os.path.exists(self.talk_file_location):
            with open(self.talk_file_location, "r") as f:
                self.itos = json.load(f)
                self.stoi = {v: int(k) for k, v in self.itos.items()}
            return

        sentence_list = preprocess_captions(sentence_list)

        frequencies = {}
        idx = 4  # idx 1, 2, 3, 0 are already taken (PAD, SOS, EOS, UNK)
        for sentence in sentence_list:
            for word in sentence.split(" "):  # self.tokenizer_eng(sentence):
                if word == "":
                    continue
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
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


class CaptioningDataset(Dataset):
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

        self.data = []
        self.captions = []

        transform = transforms.Compose(
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
                image = Image.open(img_path).convert("RGB")
                image = transform(image)

                caps = [sentence["raw"] for sentence in image_data["sentences"]]
                self.data.append(image)
                self.captions.append(caps)

        self.length = len(self.captions)
        self.vocab = Vocab(freq_threshold, dataset_name)
        self.vocab.build_vocabulary(self.text)

    def __getitem__(self, index):
        image = self.data[index]

        captions = self.captions[index]
        captions = preprocess_captions(captions)

        return image, captions

    def __len__(self):
        return self.length

    @property
    def text(self):
        all_captions = []
        for caption_list in self.captions:
            for caption in caption_list:
                all_captions.append(caption)
        return all_captions

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
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions


class Batcher:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        def sort(batch):
            batch.sort(key=lambda x: len(x[1]), reverse=True)
            return batch

        batch = sort(batch)
        images, captions = zip(*batch)

        images = torch.stack(images, 0)

        # Captions are in the form of a list of lists of strings.
        # Use the vocab to convert to a list of lists of ints,
        # adding <SOS> and <EOS> tokens.
        captions_numericalised = [
            self.vocab.numericalize(caption[0]) for caption in captions
        ]
        captions_numericalised = [
            [self.vocab.stoi["<bos>"]] + caption + [self.vocab.stoi["<eos>"]]
            for caption in captions_numericalised
        ]
        lengths = [len(caption) for caption in captions_numericalised]
        max_length = max(lengths)
        captions_numericalised = [
            caption + [self.vocab.stoi["<pad>"]] * (max_length - len(caption))
            for caption in captions_numericalised
        ]
        captions_numericalised = torch.LongTensor(captions_numericalised)

        return images, captions_numericalised, captions
