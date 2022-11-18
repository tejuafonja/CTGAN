import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ctgan import CTGAN
from ctgan import load_demo
from ctgan.experiments.utils import data_transformer
from sdmetrics.single_table.efficacy.binary import BinaryLogisticRegression
from ctgan.experiments.utils import column_metric_wrapper, histogram_intersection
from functools import partial

try:
    train_size = int(sys.argv[1])
    test_size = int(sys.argv[2])
    random_seed = int(sys.argv[3])
    metric = sys.argv[4]
    
except:
    print("Default variable fallback on line 7")
    train_size = 2000
    test_size = 10000
    random_seed = 1000
    metric = "f1"

real_data = load_demo()
train_data = real_data[:train_size]
test_data = real_data[-test_size:]


# info_loss_with_l2_cond
model_path = f"experiments/results/test_tablegan_loss/adult_{train_size}/model.pt"
ctgan_model = CTGAN(epochs=-1, verbose=True, cuda="cpu")
ctgan_model = ctgan_model.load(model_path)
ctgan_model.set_random_state(random_seed)
fake_data = ctgan_model.sample(n=test_data.shape[0])


if metric == "f1":
    transformed_test_data, transformed_fake_data, _ = data_transformer(realdata=test_data, fakedata=fake_data)

    fake_f1 = BinaryLogisticRegression.compute(test_data=transformed_test_data, 
                                            train_data=transformed_fake_data, 
                                            target="income")

    transformed_test_data, transformed_train_data, _ = data_transformer(realdata=test_data, fakedata=train_data)
    real_f1 = BinaryLogisticRegression.compute(test_data=transformed_test_data, 
                                            train_data=transformed_train_data, 
                                            target="income")

    print(f"\tTrain size: {train_size}, Test size: {test_size} \n\t Real f1: {real_f1}, \n\t Fake f1: {fake_f1}")

elif metric == "hist":
    
    func = partial(
            column_metric_wrapper,
            column_metric=partial(histogram_intersection, bins=50)
        )
    fake_hist = func(realdata=test_data, fakedata=fake_data).score.mean()
    real_hist = func(realdata=test_data, fakedata=train_data).score.mean()
    print(f"\tTrain size: {train_size}, Test size: {test_size} \n\t Real hist: {real_hist}, \n\t Fake hist: {fake_hist}")
    
else:
    raise NotImplementedError(f"{metric} is not implemented!")