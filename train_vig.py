import argparse
import itertools
import multiprocessing
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import evaluation
import factories
from evaluation import Cider, PTBTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Move to a utils file
import matplotlib.pyplot as plt
def plot_charts(train_losses, val_losses, val_scores, exp_name: str):

    val_scores_cider = [s["CIDEr"] * 100 for s in val_scores]
    val_scores_bleu1 = [s["BLEU"][0] * 100 for s in val_scores]
    val_scores_bleu4 = [s["BLEU"][3] * 100 for s in val_scores]

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title(f"Losses for {exp_name}")
    plt.savefig(f"{exp_name.replace('-', ' ').replace('_', ' ')}-losses.png")

    # clear figure
    plt.clf()
    plt.plot(val_scores_cider, label="Validation CIDEr")
    plt.plot(val_scores_bleu1, label="Validation BLEU-1")
    plt.plot(val_scores_bleu4, label="Validation BLEU-4")
    plt.legend()
    plt.title(f"Scores for {exp_name}")
    plt.savefig(f"{exp_name.replace('-', ' ').replace('_', ' ')}-evals.png")


def train_epoch_xe(model, dataloader, loss_fn, optim, scheduler, epoch, vocab):
    model.train()
    scheduler.step()
    running_loss = 0.0

    desc = "Epoch %d - train" % epoch

    with tqdm(desc=desc, unit="it", total=len(dataloader)) as pbar:
        for it, (detections, captions, _) in enumerate(dataloader):
            detections = detections.to(DEVICE)
            captions = captions.to(DEVICE)

            out = model(detections, captions)

            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()

            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()


    loss = running_loss / (it + 1)
    return loss


def train_epoch_scst(model, dataloader, optim, cider, epoch, vocab):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = 0.0
    running_reward_baseline = 0.0
    model.train()
    running_loss = 0.0
    seq_len = 20
    beam_size = 5  # TODO: make this a parameter with args

    desc = "Epoch %d - train" % epoch
    with tqdm(desc=desc, unit="it", total=len(dataloader)) as pbar:
        for it, (detections, _, caps_gt) in enumerate(dataloader):
            detections = detections.to(DEVICE)
            outs, log_probs = model.beam_search(
                detections,
                seq_len,
                vocab.vocab.stoi["<eos>"],
                beam_size,
                out_size=beam_size,
            )
            optim.zero_grad()

            # Rewards
            caps_gen = vocab.decode(outs.view(-1, seq_len))
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
                torch.from_numpy(reward).to(DEVICE).view(detections.shape[0], beam_size)
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


@torch.no_grad()
def evaluate_epoch_xe(model, dataloader, loss_fn, epoch, vocab):
    model.eval()
    running_loss = 0.0

    desc = "Epoch %d - validation" % epoch

    with tqdm(desc=desc, unit="it", total=len(dataloader)) as pbar:
        for it, (detections, captions, _) in enumerate(dataloader):
            detections = detections.to(DEVICE)
            captions = captions.to(DEVICE)

            out = model(detections, captions)
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(vocab)), captions_gt.view(-1))
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / (it + 1)
    return loss


@torch.no_grad()
def evaluate_metrics(model, dataloader, text_field, epoch):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(
        desc="Epoch %d - evaluation" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (images, enc_caps, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(DEVICE)
            # ViG encoding is called in vig_cap.step()

            with torch.no_grad():
                out, _ = model.beam_search(
                    images, 20, text_field.vocab.stoi["<eos>"], 5, out_size=1
                )
            caps_gen = text_field.decode(out, join_words=True)
            for i in range(len(caps_gt)):
                gen[f"{it}_{i}"] = [caps_gen[i]]
                gts[f"{it}_{i}"] = [caption for caption in caps_gt[i]]
            pbar.update()

    print("-" * 10)
    print(f"Predicted: {gen['0_0']}")
    print("Ground Truth:")
    print([f"{g}" for g in gts["0_0"]])
    print("-" * 10)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser(description="Train a VIG model.")
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        required=True,
        help="Dataset name [coco (default), flickr32k, flickr8k]",
    )
    parser.add_argument(
        "--dataset_img_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset images",
    )
    parser.add_argument(
        "--dataset_ann_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset annotations",
    )
    parser.add_argument(
        "--feature_limit",
        type=int,
        default=50,
        help="How many features to use per image (default: 50)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help="Name of the experiment",
    )

    # Model parameters
    parser.add_argument("--m", type=int, default=40, help="Number of memory slots")
    parser.add_argument("--n", type=int, default=3, help="Number of stacked M2 layers")
    parser.add_argument("--k", type=int, default=5, help="k for kNN graph creation")
    parser.add_argument(
        "--meshed_emb_size",
        type=int,
        default=2048,
        help="Embedding size for meshed-memory",
    )
    parser.add_argument(
        "--patch_feature_size", type=int, default=1024, help="Size of patch features"
    )
    parser.add_argument(
        "--gnn_emb_size", type=int, default=512, help="Embedding size for GNN"
    )
    parser.add_argument(
        "--sag_ratio", type=float, default=0.5, help="Ratio for SAG pooling"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=20, help="maximum epochs")
    parser.add_argument(
        "--force_rl_after", type=int, default=-1, help="force RL after (-1 to disable)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1, help="Learning rate"
    )
    parser.add_argument("--warmup", type=float, default=10000, help="Warmup steps")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of pytorch dataloader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed (-1) for no seed"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping (-1 to disable)",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    args = parser.parse_args()

    # Set random seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    # Load dataset
    train_data, val_data, test_data = factories.get_training_data(args)
    vocab = train_data.vocab
    train_dataloader = factories.get_dataloader(train_data, args.batch_size, num_workers=args.workers)
    val_dataloader = factories.get_dataloader(val_data, args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataloader = factories.get_dataloader(
        test_data, args.batch_size, shuffle=False, num_workers=args.workers
    )

    # SCST Things:
    scst_train_data, _, _ = factories.get_training_data(args)
    scst_train_dataloader = factories.get_dataloader(scst_train_data, 2, num_workers=args.workers)
    ref_caps_train = list(scst_train_data.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    # Load model
    model = factories.get_model(args, vocab).to(DEVICE)

    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Set up optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr)

    loss_fn = nn.NLLLoss(ignore_index=vocab.stoi["<pad>"])
    use_rl = False
    best_cider = 0.0
    patience = 0
    training_losses = []
    training_scores = []
    validation_losses = []
    validation_scores = []

    # Training loop
    for epoch in range(1, args.max_epochs + 1):
        if use_rl:
            training_loss, reward, reward_baseline = train_epoch_scst(
                model, scst_train_dataloader, optim, cider_train, epoch, scst_train_data
            )
            print(
                f"Epoch {epoch} - train loss: {training_loss} - reward: {reward} - reward_baseline: {reward_baseline}"
            )
        else:
            training_loss = train_epoch_xe(
                model, train_dataloader, loss_fn, optim, scheduler, epoch, vocab
            )
        training_losses.append(training_loss)

        # Validation
        with torch.no_grad():
            # scores = evaluate_metrics(model, train_dataloader, train_data, epoch)
            # training_scores.append(scores)
            # print(f"Epoch {epoch} - training scores: {scores}")
            val_loss = evaluate_epoch_xe(model, val_dataloader, loss_fn, epoch, vocab)
            validation_losses.append(val_loss)
            scores = evaluate_metrics(model, val_dataloader, train_data, epoch)
            validation_scores.append(scores)

        print(f"Epoch {epoch} - train loss: {training_loss}")
        print(f"Epoch {epoch} - validation loss: {val_loss}")
        print(f"Epoch {epoch} - validation scores: {scores}")

        cider = scores["CIDEr"]
        if cider > best_cider:
            best_cider = cider
            patience = 0
            print("Saving best model")
            torch.save(model.state_dict(), f"saved_models/{args.exp_name}-best.pt")
        else:
            patience += 1

        if patience == args.patience or (epoch == args.force_rl_after and not use_rl):
            if not use_rl and args.force_rl_after > -1:
                print("Switching to RL")
                use_rl = True

                # load best model
                model.load_state_dict(
                    torch.load(f"saved_models/{args.exp_name}-best.pt")
                )
                optim = torch.optim.Adam(model.parameters(), lr=5e-6)
            else:
                print("Early stopping")
                break

    # Load best model
    model.load_state_dict(torch.load(f"saved_models/{args.exp_name}-best.pt"))

    # Evaluate on test set
    print("*"*80)
    with torch.no_grad():
        scores = evaluate_metrics(model, test_dataloader, test_data, 0)
        print(f"Test scores: {scores}")
    print("*"*80)

    plot_charts(
        training_losses,
        validation_losses,
        validation_scores,
        args.exp_name,
    )
