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
# Fourier matrix
OpA_m = Fourier_m(mask)
# Fourier operator
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
    "num_epochs": 20000,
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
    "optimizer_params": {"lr": 5e-4, "eps": 1e-8, "weight_decay": 0},
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 100, "gamma": 0.97},
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
# sample = torch.load("/itf-fi-ml/home/johanfag/master/codebase/data/ellipses/test/sample_0.pt")
sample = torch.load("/uio/hume/student-u56/johanfag/master/codebase/data/ellipses/test/sample_0.pt")
sample = sample[None].repeat(2,1,1)
# set imaginary values to zero
sample[1] = torch.zeros_like(sample[1])
measurement = OpA(sample)
measurement = measurement.to(device)
# init noise vector (as input to the model)
noise_mag = .1
noise = noise_mag * torch.rand(sample.shape)
#noise[1] = torch.zeros_like(noise[1])
noise = noise.to(device)

# optimizer setup
optimizer        = torch.optim.Adam
scheduler        = torch.optim.lr_scheduler.StepLR
optimizer = optimizer(unet.parameters(), **train_params["optimizer_params"])
scheduler = scheduler(optimizer, **train_params["scheduler_params"])

# log setup
import pandas as pd
logging = pd.DataFrame(
            columns=["loss", "lr"]
)
# progressbar setup
from tqdm import tqdm
progress_bar = tqdm(
    desc="Train DIP ",
    total=train_params["num_epochs"],
)
from matplotlib import pyplot as plt
num_save_steps = 10
save_each = torch.ceil( torch.tensor(train_params["num_epochs"] / num_save_steps) )
save_epochs = torch.arange(train_params["num_epochs"])[::int(save_each)].tolist()# + [train_params["num_epochs"]-1]
fig, axs = plt.subplots(2,num_save_steps,figsize=(5*num_save_steps,5) )

# function that returns img of sample and the reconstructed image
def get_img_rec(sample, noise, model):
    img = torch.sqrt(sample[0]**2 + sample[1]**2).to("cpu")
    reconstruction = model.forward(noise[None])
    img_rec = torch.sqrt(reconstruction[0,0]**2 + reconstruction[0,1]**2).detach().to("cpu")
    return img, img_rec

# training loop
isave = 0
sigma_p = 1/30
for epoch in range(train_params["num_epochs"]): 
    unet.train()  # make sure we are in train mode
    optimizer.zero_grad()
    # add gaussian noise to noise input according to Ulyanov et al 2020
    additive_noise = sigma_p*torch.randn(noise.shape)
    model_input = noise + additive_noise.to(device)
    # pred = A G(z_tilde, theta)
    pred = OpA( unet.forward(model_input[None]) )
    loss = loss_func(pred, measurement[None])
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # append to log
    app_log = pd.DataFrame( {"loss": loss.item(), "lr" : scheduler.get_last_lr()[0]}, index = [0] )
    logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
    
    # update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix(
        **unet._add_to_progress_bar({"loss": loss.item()})
    )
    if epoch in save_epochs:
        img, img_rec = get_img_rec(sample, noise, model = unet)
        axs[0,isave].imshow(img_rec)
        axs[0,isave].set_title("Epoch %i"%epoch)
        axs[1,isave].imshow(.5*torch.log( (img - img_rec)**2))
        isave += 1
fig.savefig(os.getcwd() + "/DIP_evolution.png")

# save final reconstruction
#unet.eval()
img, img_rec = get_img_rec(sample, noise, model = unet) 
fig, axs = plt.subplots(1,5,figsize=(25,5) )
plot_img = axs[0].imshow(img); plot_img_rec = axs[1].imshow(img_rec);
plot_res = axs[2].imshow(torch.sqrt( (img - img_rec)**2) )
fig.colorbar(plot_img, ax = axs[0]); fig.colorbar(plot_img_rec, ax=axs[1]); fig.colorbar(plot_res, ax=axs[2])
axs[3].plot(torch.log(torch.tensor(logging["loss"])), label="log-loss", color="blue") 
axs[4].plot(logging["lr"], label="learning rate", color="orange")
fig.legend()
# savefig
save_fn = "DIP.png"
fig.savefig(os.getcwd() + "/" +  save_fn)

