import os

import numpy as np

directories = [
    "/data/scratch/eey362/superpixel_features/m25",
    "/data/scratch/eey362/superpixel_features/m50",
    "/data/scratch/eey362/superpixel_features/m75",
    "/data/scratch/eey362/superpixel_features/m100",
]

for directory in directories:
    files = os.listdir(directory)
    max_val = 0
    min_val = 99999
    total = 0
    for i, file in enumerate(files):
        # Load npz file with key "feat"
        features = np.load(os.path.join(directory, file))["feat"]
        shape = features.shape
        if shape[0] > max_val:
            max_val = shape[0]
        if shape[0] < min_val:
            min_val = shape[0]
        total += shape[0]

    print("*" * 80)
    print("Total number of features: {}".format(total))
    print("Min val for {} is {}".format(directory, min_val))
    print("Max val for {} is {}".format(directory, max_val))
    print("Average val for {} is {}".format(directory, total / len(files)))
    print("*" * 80)
    print("\n\n\n")
