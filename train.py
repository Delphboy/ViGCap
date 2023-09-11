import argparse
import itertools
import multiprocessing
import os
import pickle
import random
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import evaluation
from data import COCO, DataLoader, ImageDetectionsField, RawField, TextField
from evaluation import Cider, PTBTokenizer
from models.transformer import (
    MemoryAugmentedEncoder,
    MeshedDecoder,
    MeshedMemoryTransformer,
    ScaledDotProductAttentionMemory,
)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = 0.0
    with tqdm(
        desc="Epoch %d - validation" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools

    model.eval()
    gen = {}
    gts = {}
    with tqdm(
        desc="Epoch %d - evaluation" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                # TODO: Remove hard-coded max length. Use --flag
                # TODO: Remove hard-coded beam size. Use --flag
                out, _ = model.module.beam_search(
                    images, 20, text_field.vocab.stoi["<eos>"], 5, out_size=1
                )
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = " ".join([k for k, _ in itertools.groupby(gen_i)])
                gen["%d_%d" % (it, i)] = [
                    gen_i,
                ]
                gts["%d_%d" % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    running_loss = 0.0
    with tqdm(
        desc="Epoch %d - train" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections = detections.to(device)
            captions = captions.to(device)

            out = model(detections, captions)

            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    scheduler.step()
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = 0.0
    running_reward_baseline = 0.0
    model.train()
    running_loss = 0.0
    seq_len = 20
    beam_size = 5

    with tqdm(
        desc="Epoch %d - train" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(
                detections,
                seq_len,
                text_field.vocab.stoi["<eos>"],
                beam_size,
                out_size=beam_size,
            )
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(
                itertools.chain(
                    *(
                        [
                            c,
                        ]
                        * beam_size
                        for c in caps_gt
                    )
                )
            )
            caps_gen, caps_gt = tokenizer_pool.map(
                evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt]
            )
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = (
                torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            )
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(
                loss=running_loss / (it + 1),
                reward=running_reward / (it + 1),
                reward_baseline=running_reward_baseline / (it + 1),
            )
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description="ViGCap Training Options")
    parser.add_argument(
        "--exp_name", type=str, default="ViGCap", help="Experiment name"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--m", type=int, default=40, help="Number of meshed-memory vectors"
    )
    parser.add_argument(
        "--head", type=int, default=8, help="Number of heads in multi-head attention"
    )
    parser.add_argument(
        "--resume_last", action="store_true", help="Resume from last epoch"
    )
    parser.add_argument(
        "--resume_best", action="store_true", help="Resume from best epoch"
    )
    parser.add_argument(
        "--features_path", type=str, help="Path to COCO detection features .hdf5 file"
    )
    parser.add_argument(
        "--annotation_folder", type=str, help="Path to COCO annotation folder"
    )
    parser.add_argument(
        "--logs_folder",
        type=str,
        default="tensorboard_logs",
        help="Path to tensorboard logs folder",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random number generators"
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=-1,
        help="Run model on the test set every N epochs",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Meshed-Memory Transformer Training")

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(
        detections_path=args.features_path, max_detections=50, load_in_tmp=False
    )

    # Pipeline for text
    text_field = TextField(
        init_token="<bos>",
        eos_token="<eos>",
        lower=True,
        tokenize="spacy",
        remove_punctuation=True,
        nopoints=False,
    )

    # Create the dataset
    dataset = COCO(
        image_field,
        text_field,
        "coco/images/",
        args.annotation_folder,
        args.annotation_folder,
    )
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile("vocab.pkl"):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open("vocab.pkl", "wb"))
    else:
        text_field.vocab = pickle.load(open("vocab.pkl", "rb"))

    # Dataloaders
    # TODO: Speed this block up
    # TODO: What do these do? What are they for?
    dict_dataset_train = train_dataset.image_dictionary(
        {"image": image_field, "text": RawField()}
    )
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary(
        {"image": image_field, "text": RawField()}
    )
    dict_dataset_test = test_dataset.image_dictionary(
        {"image": image_field, "text": RawField()}
    )
    ##########

    # TODO: Do we need to hold all these dataloaders in memory?
    # What's the cost of creating them every time vs. holding them in memory?
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    dict_dataloader_train = DataLoader(
        dict_dataset_train,
        batch_size=args.batch_size // 5,
        shuffle=True,
        num_workers=args.workers,
    )
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
    dict_dataloader_test = DataLoader(
        dict_dataset_test, batch_size=args.batch_size // 5
    )
    print("Dataloaders created")

    # Build Model
    encoder = MemoryAugmentedEncoder(
        3,
        0,
        attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={"m": args.m},
    )
    decoder = MeshedDecoder(
        len(text_field.vocab), 54, 3, text_field.vocab.stoi["<pad>"]
    )
    model = MeshedMemoryTransformer(
        text_field.vocab.stoi["<bos>"], encoder, decoder
    ).to(device)
    model = nn.DataParallel(model)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Initial conditions
    optim = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.8)
    loss_fn = nn.NLLLoss(ignore_index=text_field.vocab.stoi["<pad>"])
    # loss_fn = nn.CrossEntropyLoss(ignore_index=text_field.vocab.stoi["<pad>"])
    use_rl = False
    best_cider = 0.0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = "saved_models/%s_last.pth" % args.exp_name
        else:
            fname = "saved_models/%s_best.pth" % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data["torch_rng_state"])
            torch.cuda.set_rng_state(data["cuda_rng_state"])
            np.random.set_state(data["numpy_rng_state"])
            random.setstate(data["random_rng_state"])
            model.load_state_dict(data["state_dict"], strict=False)
            optim.load_state_dict(data["optimizer"])
            scheduler.load_state_dict(data["scheduler"])
            start_epoch = data["epoch"] + 1
            best_cider = data["best_cider"]
            patience = data["patience"]
            use_rl = data["use_rl"]
            print(
                "Resuming from epoch %d, validation loss %f, and best cider %f"
                % (data["epoch"], data["val_loss"], data["best_cider"])
            )

    print("Training starts")
    for epoch in range(start_epoch, args.max_epochs):
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar("data/train_loss", train_loss, epoch)
        else:
            train_loss, reward, reward_baseline = train_scst(
                model, dict_dataloader_train, optim, cider_train, text_field
            )
            writer.add_scalar("data/train_loss", train_loss, epoch)
            writer.add_scalar("data/reward", reward, epoch)
            writer.add_scalar("data/reward_baseline", reward_baseline, epoch)

        # # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar("data/val_loss", val_loss, epoch)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores["CIDEr"]
        writer.add_scalar("data/val_cider", val_cider, epoch)
        writer.add_scalar("data/val_bleu1", scores["BLEU"][0], epoch)
        writer.add_scalar("data/val_bleu4", scores["BLEU"][3], epoch)
        writer.add_scalar("data/val_meteor", scores["METEOR"], epoch)
        writer.add_scalar("data/val_rouge", scores["ROUGE"], epoch)

        # Test scores
        if args.test_every > 0 and epoch % args.test_every == 0:
            scores = evaluate_metrics(model, dict_dataloader_test, text_field)
            print("Test scores", scores)
            writer.add_scalar("data/test_cider", scores["CIDEr"], epoch)
            writer.add_scalar("data/test_bleu1", scores["BLEU"][0], epoch)
            writer.add_scalar("data/test_bleu4", scores["BLEU"][3], epoch)
            writer.add_scalar("data/test_meteor", scores["METEOR"], epoch)
            writer.add_scalar("data/test_rouge", scores["ROUGE"], epoch)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False

        if patience == args.patience:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print("patience reached.")
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load("saved_models/%s_best.pth" % args.exp_name)
            torch.set_rng_state(data["torch_rng_state"])
            torch.cuda.set_rng_state(data["cuda_rng_state"])
            np.random.set_state(data["numpy_rng_state"])
            random.setstate(data["random_rng_state"])
            model.load_state_dict(data["state_dict"])
            print(
                "Resuming from epoch %d, validation loss %f, and best cider %f"
                % (data["epoch"], data["val_loss"], data["best_cider"])
            )

        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "random_rng_state": random.getstate(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_cider": val_cider,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "patience": patience,
                "best_cider": best_cider,
                "use_rl": use_rl,
            },
            "saved_models/%s_last.pth" % args.exp_name,
        )

        if best:
            copyfile(
                "saved_models/%s_last.pth" % args.exp_name,
                "saved_models/%s_best.pth" % args.exp_name,
            )

        if exit_train:
            writer.close()
            break
