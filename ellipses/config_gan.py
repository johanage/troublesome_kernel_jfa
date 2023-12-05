# configuration information for GANs used in these experiments
from dataclasses import dataclass
import config
from typing import Tuple, Optional, Union, List
# ----- network configuration -----
@dataclass
class Generator_params:
    img_size   : int = config.N
    latent_dim : int = 100
    # real and imag
    channels   : int = 2

@dataclass
class Discriminator_params:
    img_size : int = config.N
    # real and imag
    channels : int = 2

# ------ Definition of objective --------------
# Binary Cross Entropy between the target and the input probabilities
# BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# doc : https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
import torch
adversarial_loss_func = torch.nn.BCELoss()

# ----- measurement configuration -----
from data_management import ToComplex, SimulateMeasurements
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    RadialMaskFunc,
)

mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
OpA_m = Fourier_m(mask)
OpA = Fourier(mask)
# set device for operators
#OpA_m.to(device)

# Set optimizer parameters
@dataclass
class Optimizer_params:
    lr           : float = 1e-3
    eps          : float = 1e-8
    weight_decay : float = 0
    
# Set learning scheduler parameters
@dataclass
class Learning_scheduler_params:
    step_size : int   = 100
    gamma     : float = .98

# Set data loader parameters
@dataclass
class Dataloader_parameters:
    shuffle     : bool
    num_workers : int
# Set training parameters
import os, torchvision
@dataclass
class GAN_train_params:
    # training configuration
    save_epochs           : int = 1
    acc_steps             : int = 1
    train_phases          : int = 2
    num_epochs            : int = 100
    batch_size            : int = 40
    adversarial_loss_func : torch.nn.modules.loss.BCELoss = adversarial_loss_func
    save_path             : str = os.path.join(config.RESULTS_PATH,"DCGAN_no_jitter")
    # optimizer configurations 
    optimizer_G         : torch.optim.Adam = torch.optim.Adam
    optimizer_D         : torch.optim.Adam = torch.optim.Adam
    optimizer_G_params  : Optimizer_params = Optimizer_params()
    optimizer_D_params  : Optimizer_params = Optimizer_params()
    # learning scheduler configurations
    scheduler_G         : torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR
    scheduler_D         : torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR
    scheduler_G_params  : Learning_scheduler_params       = Learning_scheduler_params()
    scheduler_D_params  : Learning_scheduler_params       = Learning_scheduler_params()
    
    # transforms
    train_transform     : torchvision.transforms.Compose = torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)]
    )
    val_transform       : torchvision.transforms.Compose = torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)],
    )
    # dataloaders
    train_loader_params : Dataloader_parameters = Dataloader_parameters(True, 8)
    val_loader_params   : Dataloader_parameters = Dataloader_parameters(False, 8)

@dataclass
class CS_GAN_train_params:
    # training configuration
    save_epochs           : int = 1
    acc_steps             : int = 1
    num_epochs            : int = 20
    batch_size            : int = 40
    save_path             : str = os.path.join(config.RESULTS_PATH,"DCGAN_no_jitter")
    # optimizer configurations
    optimizer        : torch.optim.Adam = torch.optim.Adam
    optimizer_params : Optimizer_params = Optimizer_params()
    # learning scheduler configurations
    scheduler        : torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR
    scheduler_params : Learning_scheduler_params       = Learning_scheduler_params()
    # transforms
    train_transform     : torchvision.transforms.Compose = torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)]
    )
    val_transform       : torchvision.transforms.Compose = torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)],
    )
    # dataloaders
    train_loader_params : Dataloader_parameters = Dataloader_parameters(True, 8)
    val_loader_params   : Dataloader_parameters = Dataloader_parameters(False, 8)
