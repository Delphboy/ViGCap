import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io.image import read_image

from utils.utils import preprocess_captions


class Vocab:
    def __init__(self, freq_threshold, dataset_name="coco"):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
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
        idx = 4  # idx 0, 1, 2, 3 are already taken (PAD, SOS, EOS, UNK)
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
        tokenized_text = text.split(" ")

        val = [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
        return val


class CaptioningDataset(Dataset):
    def __init__(
        self,
        root_dir: str,  # /datasets/coco/images
        captions_file: str,  # dataset_coco.json
        dataset_name: str = "coco",
        transform: transforms.Compose = transforms.Compose(
            [transforms.functional.convert_image_dtype, transforms.Resize((224, 224))]
        ),
        freq_threshold: int = 5,
        split: str = "train",
    ):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split

        # captions_file is a json file. Load it into a dictionary
        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.data = {}
        self.captions = []

        for image in self.captions_file_data["images"]:
            if image["split"] == "restval":
                image["split"] = "train"

            if image["split"] == self.split:
                self.data[image["imgid"]] = {
                    "dir": image.get("filepath", ""),
                    "filename": image["filename"],
                    "sentences": [sentence["raw"] for sentence in image["sentences"]],
                }

            self.captions += [sentence["raw"] for sentence in image["sentences"]]

        self.ids = np.array(list(self.data.keys()))
        self.data = pd.DataFrame.from_dict(self.data, orient="index")
        self.vocab = Vocab(freq_threshold, dataset_name)
        self.vocab.build_vocabulary(self.captions)

    def __getitem__(self, index):
        data_id = self.ids[index]
        data = self.data.loc[data_id]
        captions = data["sentences"]
        captions = preprocess_captions(captions)

        image = read_image(
            os.path.join(self.root_dir, data["dir"], data["filename"]),
            ImageReadMode.RGB,
        )
        if self.transform is not None:
            image = self.transform(image)

        return image, captions[:5]

    def __len__(self):
        return len(self.ids)


class CocoBatcher:
    def __init__(self, talk_file_location):
        with open(talk_file_location, "r") as f:
            self.coco = json.load(f)

        # invert a dictionary
        self.word_to_ix = {v: k for k, v in self.coco.items()}

    def coco_ix_to_word(self, ix):
        return self.coco.ix_to_word[ix]

    def coco_word_to_ix(self, word):
        res = self.word_to_ix[word]
        return int(res)

    def captions_to_numbers(self, caption):
        numericalized_caption = []
        sanitised_caption = caption.lower().split(" ")
        for word in sanitised_caption:
            if word in self.word_to_ix:
                numericalized_caption.append(self.coco_word_to_ix(word))
            else:
                numericalized_caption.append(self.coco_word_to_ix("<UNK>"))
        return numericalized_caption

    def sorter(self, batch_element):
        captions = batch_element[1]
        lengths = [len(cap.split(" ")) for cap in captions]
        # max_length = max(lengths)
        # return max_length
        return lengths[0]

    def __call__(self, data):
        data.sort(key=self.sorter, reverse=True)
        images, captions = zip(*data)
        if max([len(x) for x in captions]) != 5:
            print("ALERT")
        images = torch.stack(images, 0)

        numericalized_captions = []
        for cap in captions:
            for caption in cap:
                numericalized_caption = [self.coco_word_to_ix("<SOS>")]
                numericalized_caption += self.captions_to_numbers(caption)
                numericalized_caption += [self.coco_word_to_ix("<EOS>")]
                tensorised = torch.tensor(numericalized_caption)
                numericalized_captions.append(tensorised)

        lengths = [len(cap) for cap in numericalized_captions]
        targets = torch.zeros(len(numericalized_captions), max(lengths)).long()

        for i, cap in enumerate(numericalized_captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, torch.tensor(lengths, dtype=torch.int64)
