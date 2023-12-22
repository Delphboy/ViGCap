import argparse
import os


def find_highest_score(dir) -> (str, float):
    highest_score = -1
    highest_score_file = None

    for root, dirs, files in os.walk(dir):
        for file in sorted(files, reverse=True):
            if file.endswith(".jpg") or file.endswith(".png"):
                continue
            with open(os.path.join(root, file), "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "validation scores" in line:
                        score_line = line.split("{")[-1]
                        score_line = "{" + score_line
                        scores = eval(score_line)
                        if scores["CIDEr"] > highest_score:
                            highest_score = scores["CIDEr"]
                            highest_score_file = file
                        break
    print(f"{highest_score_file} has highest CIDEr score of {highest_score}")
    return highest_score_file, highest_score


# Create a function like find_highest_score that instead returns all the scores but sorted by CIDEr score
# The output should be <score> \t <file name>
def find_all_scores(dir):
    scores = {}
    for root, dirs, files in os.walk(dir):
        for file in sorted(files, reverse=True):
            if file.endswith(".jpg") or file.endswith(".png"):
                continue
            with open(os.path.join(root, file), "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "Test scores" in line:
                        score_line = line.split("{")[-1]
                        score_line = "{" + score_line
                        score_line = eval(score_line)
                        scores[file] = score_line["CIDEr"]
                        break

    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for score in scores:
        print(f"{score[1]:.3f} \t {score[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find file with highest CIDEr score")
    parser.add_argument(
        "--dir", type=str, help="directory to search for files", default="logs/"
    )
    args = parser.parse_args()
    # find_highest_score(args.dir)
    find_all_scores(args.dir)
