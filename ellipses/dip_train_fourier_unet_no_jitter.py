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
    "inverter"      : None, #inverter,
}
unet = UNet
# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# set training parameters
train_params = {
    "num_epochs": 200,
    "batch_size": 1,
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            "Fourier_UNet_it_no_jitter_DIP"
        )
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 5e-5, "eps": 2e-4, "weight_decay": 1e-4},
        {"lr": 5e-5, "eps": 2e-4, "weight_decay": 1e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": 1,
}

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
measurement = measurement.to(device)
# init noise vector (as input to the model)
noise = torch.randn(sample.shape)
noise = noise.to(device)
#TODO rewrite training loop for DIP
# run training 
from tqdm import tqdm
optimizer        = torch.optim.Adam
optimizer_params = {"lr": 2e-4, "eps": 1e-3}
scheduler        = torch.optim.lr_scheduler.StepLR
scheduler_params = {"step_size": 1, "gamma": 1.0}

optimizer = optimizer(unet.parameters(), **optimizer_params)
scheduler = scheduler(optimizer, **scheduler_params)
for epoch in range(train_params["num_epochs"]):
    unet.train()  # make sure we are in train mode
    t = tqdm(
        enumerate([noise]),
        desc="epoch {} / {}".format(epoch, train_params["num_epochs"]),
        total=-(-len(noise)),
    )
    optimizer.zero_grad()
    for i, noise in t:
        #loss, inp, target, pred = unet._train_step(
        #    i, noise, loss_func, optimizer, train_params["batch_size"], train_params["acc_steps"],
        #)
        # pred = A G(z_tilde, theta)
        pred = OpA( unet.forward(noise[None]) )
        loss = loss_func(pred, measurement)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        t.set_postfix(
            **unet._add_to_progress_bar({"loss": loss.item()})
        )
