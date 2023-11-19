"""
DESCRIPTION OF WHAT THIS SCRIPT DOES
------------------------------------
Terms
    - jitter, means that noise, typically Gaussian, is added to the data while training to reduce overfit
"""


import os

import matplotlib as mpl
import torch
import torchvision

from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import UNet
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    LearnableInverterFourier,
    RadialMaskFunc,
)


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cpu")
# if GPU availablei
gpu_avail = torch.cuda.is_available()
if gpu_avail:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

# ----- measurement configuration -----
mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
OpA_m = Fourier_m(mask)
OpA = Fourier(mask)
inverter = LearnableInverterFourier(config.n, mask, learnable=False)
# set device for operators
OpA_m.to(device)
inverter.to(device)

# ----- network configuration -----
unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : inverter,
}
unet = UNet
# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )



    
train_phases = 2
train_params = {
    "num_epochs": [35, 6],
    "batch_size": [10, 10],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "Fourier_UNet_it_no_jitter_"
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 5e-5, "eps": 2e-4, "weight_decay": 1e-4},
        {"lr": 5e-5, "eps": 2e-4, "weight_decay": 1e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1, 200],
#    "train_transform": torchvision.transforms.Compose(
#        [ToComplex(), SimulateMeasurements(OpA)]
#    ),
#    "val_transform": torchvision.transforms.Compose(
#        [ToComplex(), SimulateMeasurements(OpA)],
#    ),
#    "train_loader_params": {"shuffle": True, "num_workers": 8},
#    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----

train_data_params = {
    "path": config.DATA_PATH,
}
train_data = IPDataset

val_data_params = {
    "path": config.DATA_PATH,
}
val_data = IPDataset

# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in unet_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
unet = unet(**unet_params)
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)
# get train and validation data
# data has shape (number of samples, (measurements, images) )
# Note that the second dimension consist of a 2-tuple
# image x has shape (2, N, N), since x in C^{N x N}
# measurement y has shape (2, m) since y in C^m
sample = torch.load("/itf-fi-ml/home/johanfag/master/codebase/data/ellipses/test/sample_0.pt")
sample = sample[None].repeat(2,1,1)
# set imaginary values to zero
sample[1] = torch.zeros_like(sample[1])
measurement = OpA(sample)
assert 0, "stop before training to to check status"
#train_data = train_data("train", **train_data_params)
#val_data = val_data("val", **val_data_params)

#TODO rewrite training loop for DIP
# run training 
for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))
    
    unet.train_on(train_data, val_data, **train_params_cur)
