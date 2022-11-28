import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from ctgan import CTGAN
from ctgan import load_demo

try:
    data_kind = sys.argv[1]
    train_size = int(sys.argv[2])
    sample_size = int(sys.argv[3])
except:
    print("Default variable fallback on line 7")
    data_kind = "f"
    train_size = 2000
    sample_size = 10


if data_kind == "r":
    real_data = load_demo()
    train_data = real_data[:train_size]
    print(train_data.sample(n=sample_size))

elif data_kind == "b":
    real_data = load_demo()
    train_data = real_data[:train_size]
    print(f"======real")
    print(train_data.sample(n=sample_size))

    model_path = f"experiments/results/adult_{train_size}/model.pt"
    ctgan_model = CTGAN(epochs=-1, verbose=True, cuda="cpu")
    ctgan_model = ctgan_model.load(model_path)
    synth_data = ctgan_model.sample(n=sample_size)
    print(f"======synth")
    print(synth_data)

else:
    model_path = f"experiments/results/adult_{train_size}/model.pt"
    ctgan_model = CTGAN(epochs=-1, verbose=True, cuda="cpu")
    ctgan_model = ctgan_model.load(model_path)
    synth_data = ctgan_model.sample(n=sample_size)
    print(synth_data)
