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

# vegard's implementations
from data_management import IPDataset, ToComplex, SimulateMeasurements
from networks import Generator, Discriminator
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    RadialMaskFunc,
)


# ----- load configuration -----
import config

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
# set device for operators
OpA_m.to(device)

# ----- network configuration -----
from config_gan import *
from dataclasses import asdict
generator_params = asdict(Generator_params())
discriminator_params = asdict(Discriminator_params())

# ------ Definition of objective -------------- 
# Binary Cross Entropy between the target and the input probabilities
# BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# doc : https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
adversarial_loss_func = torch.nn.BCELoss()
adversarial_loss_func.cuda()

# Set training parameters
train_params = asdict( GAN_train_params() )

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
    for key, value in generator_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in discriminator_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_params["train_phases"]) + "\n")

# ------ construct network and train -----
# TODO make the model work and make more streamlined later on
generator = Generator(**generator_params)
# parameter check holds if all params are on the same device
if next(generator.parameters()).device == torch.device("cpu"):
    generator = generator.to(device)
discriminator = Discriminator(**discriminator_params)
if next(discriminator.parameters()).device == torch.device("cpu"):
    discriminator = discriminator.to(device)
assert next(generator.parameters()).device == device and next(discriminator.parameters()).device == device, "the G and D has not yet been set to proper device"
# get train and validation data
# data has shape (number of samples, (measurements, images) )
# Note that the second dimension consist of a 2-tuple
# image x has shape (2, N, N), since x in C^{N x N}
# measurement y has shape (2, m) since y in C^m
train_data = train_data("train", **train_data_params)
val_data = val_data("val", **val_data_params)
#set transforms
train_data.transform = train_params["train_transform"]
val_data.transform = train_params["val_transform"]
# initialize data loaders
data_load_train = torch.utils.data.DataLoader(
    train_data, train_params["batch_size"], **train_params["train_loader_params"]
)
data_load_val = torch.utils.data.DataLoader(
    val_data, train_params["batch_size"], **train_params["val_loader_params"]
)

# optimizer setup
optimizer_G = train_params["optimizer_G"](generator.parameters(),     **train_params["optimizer_G_params"]) 
optimizer_D = train_params["optimizer_D"](discriminator.parameters(), **train_params["optimizer_D_params"]) 
scheduler_G = train_params["scheduler_G"](optimizer_G, **train_params["scheduler_G_params"]) 
scheduler_D = train_params["scheduler_D"](optimizer_D, **train_params["scheduler_D_params"])

# log setup
import pandas as pd
logging = pd.DataFrame(
        columns=["generator_loss", "discriminator_loss", "lr_generator", "lr_discriminator", "mem_alloc"]
)
# progressbar setup see training loop
from tqdm import tqdm

from matplotlib import pyplot as plt
num_save_steps = 5 
save_each = torch.ceil( torch.tensor(train_params["num_epochs"] / num_save_steps) )
save_epochs = torch.arange(train_params["num_epochs"])[::int(save_each)].tolist()
fig, axs = plt.subplots(2,num_save_steps,figsize=(5*num_save_steps,5) )

# load parameters
generator.load_state_dict(torch.load(train_params["save_path"] + "/generator.pth") )
discriminator.load_state_dict(torch.load(train_params["save_path"] + "/discriminator.pth") )
from trainer_gan import *
train_params["num_epoch"] = 100
generator, discriminator, logging = train_loop_gan(
    train_params          = train_params,
    generator_params      = generator_params,
    generator             = generator,
    discriminator         = discriminator,
    data_load_train       = data_load_train,
    device                = device,
    optimizer_G           = optimizer_G,
    optimizer_D           = optimizer_D,
    scheduler_G           = scheduler_G,
    scheduler_D           = scheduler_D,
    adversarial_loss_func = adversarial_loss_func,
    logging               = logging,
) 

    #TODO: make evolution plot
    
#    if epoch in save_epochs:
#        img, img_rec = get_img_rec(sample, noise, model = unet)
#        axs[0,isave].imshow(img_rec)
#        axs[0,isave].set_title("Epoch %i"%epoch)
#        axs[1,isave].imshow(.5*torch.log( (img - img_rec)**2))
#        isave += 1
#fig.savefig(os.getcwd() + "/GAN_evolution.png")

# log plot
fig, axs = plt.subplots(1,6, figsize = (25,5))
axs[0].plot(logging["generator_loss"]); axs[0].set_title("Generator loss")
axs[1].plot(logging["discriminator_loss"]); axs[1].set_title("Discriminator loss")
axs[2].plot(logging["lr_generator"], label="Generator"); axs[2].set_title("Learning schedule")
axs[2].plot(logging["lr_discriminator"], label="Discriminator"); axs[2].legend()
axs[3].plot(logging["mem_alloc"]); axs[3].set_title("Memory consumption")
z = torch.randn(1, generator_params["latent_dim"]).to(device)
generated_image = generator.forward(z).to("cpu").detach()
axs[4].imshow( (generated_image[0,0]**2 + generated_image[0,1]**2)**.5 )
axs[4].set_title("Generated image")
image = (data_load_train.dataset[0][1][0]**2 + data_load_train.dataset[0][1][1]**2)**.5 
axs[5].imshow(image)
axs[5].set_title("Example real image")
fig.savefig(os.getcwd() + "/GAN_test_log.png")
# save final reconstruction
# TODO: get image and image reconstruction from complex vectors
# TODO: make log-plot
#img, img_rec = get_img_rec(sample, noise, model = unet) 
#fig, axs = plt.subplots(1,5,figsize=(25,5) )
#plot_img = axs[0].imshow(img); plot_img_rec = axs[1].imshow(img_rec);
#plot_res = axs[2].imshow(torch.sqrt( (img - img_rec)**2) )
#fig.colorbar(plot_img, ax = axs[0]); fig.colorbar(plot_img_rec, ax=axs[1]); fig.colorbar(plot_res, ax=axs[2])
#axs[3].plot(torch.log(torch.tensor(logging["loss"])), label="log-loss", color="blue") 
#axs[4].plot(logging["lr"], label="learning rate", color="orange")
#fig.legend()
# savefig
#save_fn = "GAN.png"
#fig.savefig(os.getcwd() + "/" +  save_fn)

## TODO: implement optimization over latent vector z
