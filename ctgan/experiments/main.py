import sys

from ctgan import CTGAN
from ctgan import load_demo

import random
import torch
import numpy as np

import os
import shutil


real_data = load_demo()

try:
    train_size = int(sys.argv[1])
    max_subset = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    random_seed = int(sys.argv[4])

except:
    print("Default variable fallback on line 7")
    train_size = 2000
    max_subset = 2000
    n_epochs = 1000
    random_seed = 1000

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


log_dir = f"experiments/results/info_loss_sdgym/l1/adult_{train_size}"
os.makedirs(log_dir, exist_ok=True)

train_data = real_data[:train_size]

ep = min((max_subset * n_epochs) // train_size, 5000)

print(
    f"\tTrain size: {train_size} \n\t Maximum subset: {max_subset}"
    f"\n\t n_epochs: {n_epochs}"
)

generator_penalty_dict = {"loss": "info_loss_sdgym", "norm_order": 1}

evaluation_dict = {"target": "income", "bins": [5, 20, 50]}
plot_title = (f"Data=Adult_{train_size}, " 
             f"Loss={generator_penalty_dict['loss']}, "
             f"norm=l{generator_penalty_dict['norm_order']}")

log_dict = {"log_dir": log_dir, "title": "ctgan", "plot_title": plot_title}

ctgan_model = CTGAN(
    epochs=ep,
    verbose=True,
    cuda="cpu",
    log_dict=log_dict,
    generator_penalty_dict=generator_penalty_dict,
    evaluation_dict=evaluation_dict,
)

ctgan_model.fit(train_data, discrete_columns)

## Store this script
shutil.copy(os.path.realpath(__file__), log_dir)