import argparse
import multiprocessing
import os
import random
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import evaluation
import factories
from evaluation import Cider, PTBTokenizer


def evaluate_loss(model, dataloader, loss_fn):
    # Validation loss
    model.eval()
    running_loss = 0.0
    with tqdm(
        desc="Epoch %d - validation" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        with torch.no_grad():
            for it, (detections, captions, lengths) in enumerate(dataloader):
                detections = detections.to(device)
                captions = captions[0:-1:5, :].to(device)

                out = model(detections, captions)

                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(
                    out.view(-1, len(dataloader.dataset.vocab)), captions.view(-1)
                )
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    model.eval()
    gen = {}
    gts = {}

    with tqdm(
        desc="Epoch %d - evaluation" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (images, caps_gt, lengths) in enumerate(iter(dataloader)):
            images = images.to(device)

            with torch.no_grad():
                # TODO: Remove hard-coded max length. Use --flag
                # TODO: Remove hard-coded beam size. Use --flag
                out, _ = model.module.beam_search(
                    model.module.vig(images),
                    20,
                    text_field.vocab.stoi["<EOS>"],
                    5,
                    out_size=1,
                )
            caps_gen = []
            for b in range(out.shape[0]):
                caps_gen.append(
                    " ".join(
                        [
                            dataloader.dataset.vocab.itos[str(int(i))]
                            for i in out[b]
                            if i != text_field.vocab.stoi["<PAD>"]
                            and i != text_field.vocab.stoi["<EOS>"]
                            and i != text_field.vocab.stoi["<SOS>"]
                        ]
                    )
                )

            caps_gt = [
                " ".join(
                    [
                        dataloader.dataset.vocab.itos[str(int(i))]
                        for i in cap
                        if i != text_field.vocab.stoi["<PAD>"]
                        and i != text_field.vocab.stoi["<EOS>"]
                        and i != text_field.vocab.stoi["<SOS>"]
                    ]
                )
                for cap in caps_gt
            ]

            caps_gt = [caps_gt[i : i + 5] for i in range(0, len(caps_gt), 5)]

            if args.debug:
                print(caps_gen)
                print(caps_gt)
                print()

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen["%d_%d" % (it, i)] = [
                    gen_i,
                ]
                gts["%d_%d" % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim):
    # Training with cross-entropy
    model.train()
    running_loss = 0.0
    with tqdm(
        desc="Epoch %d - train" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (images, captions, lengths) in enumerate(dataloader):
            images = images.to(device)
            captions = captions[0:-1:5, :].to(device)

            out = model(images, captions)

            optim.zero_grad(set_to_none=True)
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(
                out.view(-1, len(dataloader.dataset.vocab)), captions_gt.view(-1)
            )
            loss.backward()

            optim.step()
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    # scheduler.step()
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = 0.0
    running_reward_baseline = 0.0
    model.train()
    running_loss = 0.0
    seq_len = 20
    beam_size = 3
    out_size = 1

    with tqdm(
        desc="Epoch %d - train" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (images, caps_gt, lengths) in enumerate(dataloader):
            images = images.to(device)
            outs, log_probs = model.module.beam_search(
                model.module.vig(images),
                seq_len,
                text_field.vocab.stoi["<EOS>"],
                beam_size,
                out_size=out_size,
            )
            optim.zero_grad()

            # Rewards
            caps_gen = []
            for b in range(outs.shape[0]):
                caps_gen.append(
                    " ".join(
                        [
                            dataloader.dataset.vocab.itos[str(int(i))]
                            for i in outs[b]
                            if i != text_field.vocab.stoi["<PAD>"]
                            and i != text_field.vocab.stoi["<EOS>"]
                            and i != text_field.vocab.stoi["<SOS>"]
                        ]
                    )
                )

            caps_gt = [
                " ".join(
                    [
                        dataloader.dataset.vocab.itos[str(int(i))]
                        for i in cap
                        if i != text_field.vocab.stoi["<PAD>"]
                        and i != text_field.vocab.stoi["<EOS>"]
                        and i != text_field.vocab.stoi["<SOS>"]
                    ]
                )
                for cap in caps_gt
            ]

            caps_gt = [caps_gt[i : i + 5] for i in range(0, len(caps_gt), 5)]

            caps_gen, caps_gt = tokenizer_pool.map(
                evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt]
            )
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(images.shape[0], out_size)
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
        "--dataset",
        type=str,
        default="coco",
        help="[coco (default) | flickr30k | flickr8k]",
    )
    parser.add_argument(
        "--dataset_img_path",
        type=str,
        help="Path to the dataset images folder",
    )
    parser.add_argument(
        "--dataset_ann_path",
        type=str,
        help="Path to the dataset annotations file",
    )
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
        "--patience", type=int, default=5, help="Early stopping patience | -1 = disable"
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
        "--logs_folder",
        type=str,
        default="tensorboard_logs",
        help="Path to tensorboard logs folder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for random number generators. -1 for random seed each time",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=-1,
        help="Run model on the test set every N epochs",
    )
    parser.add_argument(
        "--vig_size",
        type=str,
        default="tiny",
        help="ViG model size [tiny | small | base]",
    )
    parser.add_argument(
        "--vig_type",
        type=str,
        default="default",
        help="ViG model type [default | pyramid]",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: enable pytorch debugging APIs",
    )

    args = parser.parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not args.debug:
        # disable debugging APIs
        # any PyTorch APIs are intended for debugging and should be disabled for regular
        # training runs:
        # ie)
        # anomaly detection: torch.autograd.detect_anomaly or torch.autograd.set_detect_anomaly(True)
        # profiler related: torch.autograd.profiler.emit_nvtx, torch.autograd.profiler.profile
        # autograd gradcheck: torch.autograd.gradcheck or torch.autograd.gradgradcheck

        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)
    # TODO: disable gradcheck
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-debugging-apis

    print("ViGCap Training")

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Load the data
    train_data, val_data, test_data = factories.get_training_data(args)
    train_dataloader = factories.get_dataloader(
        train_data, args.dataset, args.batch_size, True
    )
    val_dataloader = factories.get_dataloader(
        val_data, args.dataset, args.batch_size, False
    )
    test_dataloader = factories.get_dataloader(
        test_data, args.dataset, args.batch_size, False
    )

    ref_caps_train = list(train_data.captions)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    print("Dataloaders created")

    model = factories.get_model(args, train_data.vocab).to(device)
    model = nn.DataParallel(model)

    # Initial conditions
    optim = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.8)
    # loss_fn = nn.NLLLoss(ignore_index=train_data.vocab.stoi["<PAD>"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_data.vocab.stoi["<PAD>"])

    use_rl = False
    best_cider = 0.0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = "saved_models/%s_last.pth" % args.exp_name
            fname = "saved_models/debug_last.pth"
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
        use_rl = epoch == 2
        if not use_rl:
            train_loss = train_xe(model, train_dataloader, optim)
            writer.add_scalar("data/train_loss", train_loss, epoch)
        else:
            train_dataloader = factories.get_dataloader(
                train_data, args.dataset, args.batch_size // 32, True
            )
            train_loss, reward, reward_baseline = train_scst(
                model, train_dataloader, optim, cider_train, train_data
            )
            writer.add_scalar("data/train_loss", train_loss, epoch)
            writer.add_scalar("data/reward", reward, epoch)
            writer.add_scalar("data/reward_baseline", reward_baseline, epoch)

        # Validation loss
        val_loss = evaluate_loss(model, val_dataloader, loss_fn)
        writer.add_scalar("data/val_loss", val_loss, epoch)

        # Validation scores
        scores = evaluate_metrics(model, val_dataloader, val_data)
        print("Validation scores", scores)
        val_cider = scores["CIDEr"]
        writer.add_scalar("data/val_cider", val_cider, epoch)
        writer.add_scalar("data/val_bleu1", scores["BLEU"][0], epoch)
        writer.add_scalar("data/val_bleu4", scores["BLEU"][3], epoch)
        writer.add_scalar("data/val_meteor", scores["METEOR"], epoch)
        writer.add_scalar("data/val_rouge", scores["ROUGE"], epoch)

        # Test scores
        if args.test_every > 0 and epoch % args.test_every == 0:
            scores = evaluate_metrics(model, test_dataloader, test_data)
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
