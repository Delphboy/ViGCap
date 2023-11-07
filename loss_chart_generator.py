import argparse
import os

import matplotlib.pyplot as plt


def read_file_until_rl(path: str, is_rl: bool) -> list:
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if is_rl and line == "Switching to RL\n":
                return lines[:i]
    return lines


def extract_loss_values(lines: list, search_str: str, every: int = 18) -> list:
    losses = []
    for line in lines:
        if search_str in line:
            split_line = line.split("=")
            loss = split_line[-1][:-2]
            loss = float(loss.strip("]"))
            if loss != loss:
                print("Warning: NaN loss detected")
            losses.append(loss)
    return losses[::every]


def extract_validation_scores(lines: list) -> list:
    # search for line with "Validation scores"
    # split it on {
    scores = []
    for line in lines:
        if "validation scores" in line:
            split_line = line.split("{")
            score = "{" + split_line[-1][:-2] + "}"
            score_dict = eval(score)
            scores.append(score_dict)
    return scores


def extract_training_loss_values(lines: list) -> list:
    every = len([i for i, line in enumerate(lines) if "Epoch 1 - train: 100%" in line])
    return extract_loss_values(lines, "- train: 100%", every=every)


def extract_validation_loss_values(lines: list) -> list:
    every = len(
        [i for i, line in enumerate(lines) if "Epoch 1 - evaluate: 100%" in line]
    )
    return extract_loss_values(lines, "- evaluate: 100%", every=every)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--file", type=str)
    args.add_argument("--rl", action="store_true")
    args.add_argument("--dir", type=str)

    args = args.parse_args()

    files = []

    if args.dir is None:
        files.append(args.file)
    else:
        for file in os.listdir(args.dir):
            files.append(os.path.join(args.dir, file))

    for file in files:
        file_lines = read_file_until_rl(file, args.rl)

        training_losses = extract_training_loss_values(file_lines)
        validation_losses = extract_validation_loss_values(file_lines)
        validation_scores = extract_validation_scores(file_lines)
        cider_scores = [score["CIDEr"] for score in validation_scores]
        b4_scores = [score["BLEU"][3] for score in validation_scores]

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(training_losses, label="Training Loss")
        ax1.plot(validation_losses, label="Validation Loss")
        ax1.plot(cider_scores, label="CIDEr Score")
        ax1.plot(b4_scores, label="B4 Score")
        ax1.legend()
        plt.title("Training and Validation Losses for XE training")

        # save to args.file + ".png"
        plt.savefig(file + ".png")
