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
# Fourier matrix
OpA_m = Fourier_m(mask)
# Fourier operator
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
gan_train_params = GAN_train_params()
gan_train_params.num_epochs = 0
train_params = asdict( gan_train_params )

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
#from tqdm import tqdm

#from matplotlib import pyplot as plt
#num_save_steps = 5 
#save_each = torch.ceil( torch.tensor(train_params["num_epochs"] / num_save_steps) )
#save_epochs = torch.arange(train_params["num_epochs"])[::int(save_each)].tolist()
#fig, axs = plt.subplots(2,num_save_steps,figsize=(5*num_save_steps,5) )

# load parameters
generator.load_state_dict(torch.load(train_params["save_path"] + "/generator_no_jitter_epoch90.pth") )
discriminator.load_state_dict(torch.load(train_params["save_path"] + "/discriminator_no_jitter_epoch90.pth") )
from importlib import reload
import trainer_gan; reload(trainer_gan)
from trainer_gan import *
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
    jitter                = False,
    fn_suffix                = "_no_jitter"
) 

# TODO: implement optimization over latent vector z
# log z optimisation
logging_zoptim = pd.DataFrame(
        columns=["objective", "measurement_error", "representation_error", "lr", "mem_alloc"]
)

# load CS train parameters
cs_gan_train_params = asdict( CS_GAN_train_params() )

# init generator input
init_shape = (cs_gan_train_params["batch_size"], generator_params["latent_dim"])
z_init = torch.randn(init_shape).to(device)

# optimize generator input
#breakpoint()
z_optim, logging = optimize_generator_input_vector(
    cs_train_params      = cs_gan_train_params,
    z_init               = z_init,
    generator            = generator,
    data_load_train      = data_load_train,
    measurement_operator = OpA,
    logging              = logging_zoptim,
    device               = device, 
)

