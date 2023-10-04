import os

VIG_SIZES = ['tiny', 'small', 'base']
VIG_TYPES = ['default']
BLOCK_SIZES = [i for i in range(1, 17)]

# The file name format is: block_{blocks}_flickr8k_VigCap-{vig_type}-{vigsize}.out
# Example: block_16_flickr8k_VigCap-default-base.out
# Build a list of all possible file names
FILES = [f"block_{block}_flickr8k_ViGCap-{vig_type}-{vigsize}" for block in BLOCK_SIZES for vig_type in VIG_TYPES for vigsize in VIG_SIZES]

def get_parameters_from_file_name(file_name: str) -> list:
    split = file_name.split('-')
    block = int(split[0].split('_')[1])
    vig_size = split[2].split('.')[0]
    return block, vig_size


def get_final_validation_metrics_from_file_dir(file_dir: str) -> dict:
    # if the file doesn't exist return an empty dict
    if not os.path.exists(file_dir):
        return {'BLEU': [0,0,0,0],
                'METEOR': 0,
                'ROUGE': 0,
                'CIDEr': 0}

    with open(file_dir) as f:
        lines = f.readlines()
        last_line = lines[-1]
    
    score = last_line.split('{')[-1]
    scores = eval('{' + score)
    return scores

if __name__ == "__main__":
    print("Block\tVigSize\tBLEU-4\tCIDEr")
    print("=" * 30)
    for file in FILES:
        block, vig_size = get_parameters_from_file_name(file)
        results = get_final_validation_metrics_from_file_dir(os.path.join('logs', file + '.out'))
        print(f"{block}\t{vig_size}\t{results['BLEU'][3]:.3f}\t{results['CIDEr']:.3f}")
        print("-" * 30) if vig_size == 'base' else None