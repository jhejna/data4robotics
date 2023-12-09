import glob
import os
import numpy as np
import json


INPUT_PATH  = "/iliad/u/jhejna/datasets/orca/blue_mug/train"
OUTPUT_PATH = "/iliad/u/jhejna/data4robotics/blue_mug_dataset_statistics.json"

all_actions = []

episodes = glob.glob(os.path.join(INPUT_PATH, "*.npz"))
for episode in episodes:
    with open(episode, "rb") as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    all_actions.append(data["action"][1:])

all_actions = np.concatenate(all_actions, axis=0)
mean = all_actions.mean(axis=0)
std = all_actions.std(axis=0)

# save these values
with open(OUTPUT_PATH, 'w') as json_file:
    json.dump(dict(mean=mean.tolist(), std=std.tolist()), json_file)
