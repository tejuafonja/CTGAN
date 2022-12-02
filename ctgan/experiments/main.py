import sys

from ctgan import CTGAN
from ctgan import load_demo

import random
import torch
import numpy as np
import pandas as pd

import os
import shutil

import os

switch_to = os.environ.get("switch_to")
print(switch_to)


try:
    train_size = int(sys.argv[1])
    max_subset = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    random_seed = int(sys.argv[4])
    folder_name = sys.argv[5]
    disable_condvec = bool(int(sys.argv[6]))
    dataset = sys.argv[7]
    verbose = bool(int(sys.argv[8]))

except:
    print("Default variable fallback on line 7")
    train_size = 2000
    max_subset = 2000
    n_epochs = 1000
    random_seed = 1000
    folder_name = ""
    disable_condvec = False
    dataset = "adult"
    verbose = False

# Set random seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)



if dataset == "adult":
    real_data = load_demo()

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
    evaluation_dict = {"target": "income"}
    
elif dataset == "texas":
    real_data = pd.read_csv("./experiments/data/texas/train.csv", index_col=None)
    discrete_columns = [
        "DISCHARGE",
        "PAT_STATE",
        "SEX_CODE",
        "TYPE_OF_ADMISSION",
        "PAT_STATUS",
        "RACE",
        "ADMIT_WEEKDAY",
        "ETHNICITY",
        "PAT_AGE",
        "ILLNESS_SEVERITY",
        "RISK_MORTALITY",
    ]
    num_cols = [
        "LENGTH_OF_STAY",
        "TOTAL_CHARGES",
        "TOTAL_NON_COV_CHARGES",
        "TOTAL_CHARGES_ACCOMM",
        "TOTAL_NON_COV_CHARGES_ACCOMM",
        "TOTAL_CHARGES_ANCIL",
        "TOTAL_NON_COV_CHARGES_ANCIL",
    ]
    real_data.loc[:, discrete_columns] = real_data.loc[:, discrete_columns].astype("object")
    real_data.loc[:, num_cols] = real_data.loc[:, num_cols].astype("float")
    evaluation_dict = {"target": "RISK_MORTALITY"}
else:
    raise NotImplementedError()

train_data = real_data[:train_size]
ep = min((max_subset * n_epochs) // train_size, 5000)
# ep=300

print(
    f"\tDataset: {dataset}"
    f"\tTrain size: {train_size} \n\t Maximum subset: {max_subset}"
    f"\n\t n_epochs: {n_epochs} \n\t ep: {ep} \n\t folder: {folder_name}"
    f"\n\t Disable Condvec: {disable_condvec} \n\t Verbose: {verbose}"
)


if folder_name.split("/")[-1] == "baseline":
    generator_penalty_dict = None
    plot_title = f"Data={dataset}_{train_size}, Loss=baseline"
else:
    losses = folder_name.split("/")[-1].split("+")
    generator_penalty_dict = {
        "loss": losses,
    }
    plot_title = f"Data={dataset}_{train_size}, Loss={losses}"

print(generator_penalty_dict)

log_dir = f"experiments/results/{folder_name}/{dataset}_{train_size}"
os.makedirs(log_dir, exist_ok=True)

## Store this script
shutil.copy(os.path.realpath(__file__), log_dir)


log_dict = {"log_dir": log_dir, "title": "ctgan", "plot_title": plot_title}

if switch_to == "modified":
    ctgan_model = CTGAN(
        epochs=ep,
        verbose=verbose,
        cuda="cpu",
        disable_condvec=disable_condvec,
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
