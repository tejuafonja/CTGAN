import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from ctgan import CTGAN
from ctgan import load_demo
from ctgan.experiments.utils import data_transformer
from sdmetrics.single_table.efficacy.binary import BinaryLogisticRegression
from ctgan.experiments.utils import column_metric_wrapper, histogram_intersection
from functools import partial

import random
import numpy as np
import torch

try:
    train_size = int(sys.argv[1])
    test_size = int(sys.argv[2])
    random_seed = int(sys.argv[3])
    metric = sys.argv[4]
    model_name = sys.argv[5]

except:
    print("Default variable fallback on line 7")
    train_size = 2000
    test_size = 10000
    random_seed = 1000
    metric = "f1"
    model_name = ""

real_data = load_demo()
train_data = real_data[:train_size]
test_data = real_data[-test_size:]


# Set random seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# info_loss_with_l2_cond
model_path = f"experiments/results/{model_name}/adult_{train_size}/model.pt"
print(model_path)

ctgan_model = CTGAN(epochs=-1, verbose=True, cuda="cpu")
ctgan_model = ctgan_model.load(model_path)
ctgan_model.set_random_state(random_seed)
fake_data = ctgan_model.sample(n=test_data.shape[0])

fake_f1 = 0
real_f1 = 0
fake_hist = 0
real_hist = 0

for m in metric.split("+"):
    if m == "f1":
        transformed_test_data, transformed_fake_data, _ = data_transformer(
            realdata=test_data, fakedata=fake_data
        )

        fake_f1 = BinaryLogisticRegression.compute(
            test_data=transformed_test_data,
            train_data=transformed_fake_data,
            target="income",
        )

        transformed_test_data, transformed_train_data, _ = data_transformer(
            realdata=test_data, fakedata=train_data
        )
        real_f1 = BinaryLogisticRegression.compute(
            test_data=transformed_test_data,
            train_data=transformed_train_data,
            target="income",
        )

        print(
            f"\tTrain size: {train_size}, Test size: {test_size} \n\t Real f1: {fake_f1}, \n\t Fake f1: {real_f1}"
        )

    elif m == "hist":

        func = partial(
            column_metric_wrapper,
            column_metric=partial(histogram_intersection, bins=50),
        )
        fake_hist = func(realdata=test_data, fakedata=fake_data).score.mean()
        real_hist = func(realdata=test_data, fakedata=train_data).score.mean()
        print(
            f"\tTrain size: {train_size}, Test size: {test_size} \n\t Real hist: {real_hist}, \n\t Fake hist: {fake_hist}"
        )

    else:
        raise NotImplementedError(f"{m} is not implemented!")


with open(f"experiments/results/metric_result.txt", "a") as f:
    f.write(
        f"model_name={model_name}\nmetric={metric}\ntrain_size/test_size={train_size}/{test_size}"
        f"\nreal_f1/real_hist={real_f1*100:.2f} / {real_hist*100:.2f}"
        f"\nfake_f1/fake_hist={fake_f1*100:.2f} / {fake_hist*100:.2f}\n"
    )
    f.write("==========\n")
