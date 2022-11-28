import sys

from ctgan import CTGAN
from ctgan import load_demo

import random
import torch
import numpy as np

import os
import shutil

import os

switch_to = os.environ.get("switch_to")
print(switch_to)


real_data = load_demo()

try:
    train_size = int(sys.argv[1])
    max_subset = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    random_seed = int(sys.argv[4])
    folder_name = sys.argv[5]

except:
    print("Default variable fallback on line 7")
    train_size = 2000
    max_subset = 2000
    n_epochs = 1000
    random_seed = 1000
    folder_name = ""

# Set random seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Names of the columns that are discrete
discrete_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]


train_data = real_data[:train_size]
ep = min((max_subset * n_epochs) // train_size, 5000)

print(
    f"\tTrain size: {train_size} \n\t Maximum subset: {max_subset}"
    f"\n\t n_epochs: {n_epochs}"
)


verbose = False
if folder_name.split("/")[-1] == "baseline":
    generator_penalty_dict = None
    plot_title = f"Data=Adult_{train_size}, Loss=baseline"
else:
    losses = folder_name.split("/")[-1].split("+")
    generator_penalty_dict = {
        "loss": losses,
    }
    plot_title = f"Data=Adult_{train_size}, Loss={losses}"

print(generator_penalty_dict)

log_dir = f"experiments/results/{folder_name}/adult_{train_size}"
os.makedirs(log_dir, exist_ok=True)

## Store this script
shutil.copy(os.path.realpath(__file__), log_dir)


evaluation_dict = {"target": "income", "bins": [5, 20, 50]}
log_dict = {"log_dir": log_dir, "title": "ctgan", "plot_title": plot_title}

if switch_to == "modified":
    ctgan_model = CTGAN(
        epochs=ep,
        verbose=verbose,
        cuda="cpu",
        log_dict=log_dict,
        generator_penalty_dict=generator_penalty_dict,
        evaluation_dict=evaluation_dict,
    )
    ctgan_model.fit(train_data, discrete_columns)

    if not verbose:
        ctgan_model.save(f"{log_dir}/model.pt")
else:
    ctgan_model = CTGAN(
        epochs=ep,
        verbose=verbose,
        cuda="cpu",
    )
    ctgan_model.fit(train_data, discrete_columns)
    ctgan_model.save(f"{log_dir}/model.pt")
