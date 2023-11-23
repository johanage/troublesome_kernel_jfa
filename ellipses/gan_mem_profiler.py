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
import config  # isort:skip

# Enable memory profiler
from torch.profiler import profile, ProfilerActivity, record_function
with profile(
    activities=[ProfilerActivity.CUDA], 
    profile_memory=True, 
    record_shapes=True
) as prof:

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
    generator_params = {
        "img_size"   : config.N,
        "latent_dim" : int(5e-3*config.N**2),
        # real and imag 
        "channels"   : 2,
    }
    discriminator_params = {
        "img_size" : config.N,
        # real and imag 
        "channels" : 2,
    }

    # ------ definition of objective -------------- 
    #def adversarial_loss_func():
    #    pass
    # Binary Cross Entropy between the target and the input probabilities
    # BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    # doc : https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    adversarial_loss_func = torch.nn.BCELoss()
    adversarial_loss_func.cuda()
    # set training parameters
    train_params = {
        "num_epochs": 50,
        "batch_size": 100,
        "adversarial_loss_func": adversarial_loss_func,
        "save_path": [
            os.path.join(
                config.RESULTS_PATH,
                "Fourier_UNet_it_no_jitter_DIP"
            )
        ],
        "save_epochs": 1,
        "optimizer_G": torch.optim.Adam,
        "optimizer_D": torch.optim.Adam,
        "optimizer_G_params": {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0},
        "optimizer_D_params": {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0},
        "scheduler_G": torch.optim.lr_scheduler.StepLR,
        "scheduler_D": torch.optim.lr_scheduler.StepLR,
        "scheduler_G_params": {"step_size": 100, "gamma": .98},
        "scheduler_D_params": {"step_size": 100, "gamma": .98},
        "acc_steps": 1,
        "train_phases" : 2,
        "train_transform": torchvision.transforms.Compose(
            [ToComplex(), SimulateMeasurements(OpA)]
        ),
        "val_transform": torchvision.transforms.Compose(
            [ToComplex(), SimulateMeasurements(OpA)],
        ),
        "train_loader_params": {"shuffle": True, "num_workers": 8},
        "val_loader_params": {"shuffle": False, "num_workers": 8},
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
    num_save_steps = 10
    save_each = torch.ceil( torch.tensor(train_params["num_epochs"] / num_save_steps) )
    save_epochs = torch.arange(train_params["num_epochs"])[::int(save_each)].tolist()
    fig, axs = plt.subplots(2,num_save_steps,figsize=(5*num_save_steps,5) )

    # TODO: training loop
    isave = 0
    for epoch in range(1):#range(train_params["num_epochs"]): 
        # make sure we are in train mode
        generator.train()  
        discriminator.train()
        progress_bar = tqdm(
            enumerate(data_load_train),
            desc="Train GAN epoch %i"%epoch,
            total=len(train_data)//train_params["batch_size"],
        )
        for i, batch in progress_bar: 
            measurements, images = batch
            measurments = measurements.to(device); images = images.to(device)
            # -----------------------------------------------------------
            # Train generator
            # -----------------------------------------------------------
            optimizer_G.zero_grad() 
            # draw random latent vector z
            latent_vector = torch.randn(images.shape[0], generator_params["latent_dim"]).to(device)
            generated_images = generator.forward(latent_vector).to(device)
            assert generated_images.shape == images.shape, "generated vector does not have the same shape as images - update your training parameters"
            discriminated_gen_imgs = discriminator.forward(generated_images).to(device)
            generator_loss = adversarial_loss_func(discriminated_gen_imgs, torch.ones_like(discriminated_gen_imgs) )
            generator_loss.backward()
            optimizer_G.step()
            scheduler_G.step()
            # ----------------------------------------------------------
            # Train discriminator
            # ----------------------------------------------------------
            optimizer_D.zero_grad()
            # measure discriminator's ability to distinguish real from generated images
            discriminated_imgs = discriminator.forward(images)
            real_loss = adversarial_loss_func( discriminated_imgs, torch.ones_like(discriminated_imgs) )
            fake_loss = adversarial_loss_func( discriminated_gen_imgs, torch.zeros_like(discriminated_gen_imgs) )
            discriminator_loss = (real_loss + fake_loss)/2
            # append to log
            app_log = pd.DataFrame( 
                {
                "generator_loss"     : generator_loss.item(), 
                "discriminator_loss" : discriminator_loss.item(), 
                "lr_generator"       : scheduler_G.get_last_lr()[0],
                "lr_discriminator"   : scheduler_D.get_last_lr()[0],
                "mem_alloc"          : torch.cuda.memory_allocated(),
                }, 
                index = [0] )
            logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
            
            # update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(**{
                "gen_loss"   : generator_loss.item(),
                "discr_loss" : discriminator_loss.item(),
            }
            )
print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
