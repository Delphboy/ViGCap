import os
import argparse

VIG_TYPES = ['pyramid', 'default']
MODEL_TYPES = ['tiny', 'small', 'base']

COMBOS = [f"{vig}-{model}" for vig in VIG_TYPES for model in MODEL_TYPES]


# get files in a given directory whose file name contains a given string
# and the file type is .out
def get_files(dir: str, search_str: str) -> list:
    files = []
    for file in os.listdir(dir):
        if file.startswith(search_str) and file.endswith(".out"):
            files.append(file)
    return files


def get_final_cider_from_file_dir(file_dir: str) -> float:
    # Read the last line of the file
    with open(file_dir) as f:
        lines = f.readlines()
        last_line = lines[-1]
    score = last_line.split('{')[-1]
    scores = eval('{' + score)
    return scores


def print_table(scores: dict) -> None:
    print("Model\t\tB-4\tMETEOR\tROUGE\tCIDEr")
    print("-" * 50)
    for combo in COMBOS:
        # get the key from scores that contains the substring combo
        key = [key for key in scores.keys() if combo in key][0]
        score = scores[key]
        print(f"{combo}\t{score['BLEU'][3]:.3f}\t{score['METEOR']:.3f}\t{score['ROUGE']:.3f}\t{score['CIDEr']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', type=str, required=True, help='search string')
    args = parser.parse_args()
    dir = 'logs'
    files = get_files(dir, args.search)
    scores = {file: get_final_cider_from_file_dir(os.path.join(dir, file)) for file in files}
    print_table(scores)

    