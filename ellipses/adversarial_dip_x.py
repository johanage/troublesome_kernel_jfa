from matplotlib import pyplot as plt
import os, sys, torch, numpy as np
from networks import UNet
from operators import (
    Fourier,
    Fourier_matrix as Fourier_m,
    RadialMaskFunc,
    MaskFromFile,
)
import config
from operators import to_complex
# set device
gpu_avail = torch.cuda.is_available()
if gpu_avail:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

mask_fromfile = MaskFromFile(
    path = os.getcwd() + "/sampling_patterns/",
    filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png"
)
# Fourier matrix
OpA = Fourier(mask_fromfile.mask[None])
OpA_m = Fourier_m(mask_fromfile.mask[None])

# set device for operators
OpA_m.to(device)

# ----- Network configuration -----
unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : None, 
}
unet = UNet
# ------ construct network and train -----
unet = unet(**unet_params)
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# get train and validation data
dir_train = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/train/"
dir_val = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/val/"

# same as DIP
tar = torch.load(dir_train + "sample_00000.pt").to(device)
tar_complex = to_complex(tar[None]).to(device)
measurement = OpA(tar_complex).to(device)[None]

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
noise_mag = .1
# same shape as SR problem in Ulyanov et al 2018
#random init : z_tilde = noise_mag * torch.rand((unet_params["in_channels"],) + tar.shape)
z_tilde = torch.load(os.getcwd() + "/adv_attack_dip/z_tilde.pt")
z_tilde = z_tilde.to(device)

# load model weights
param_dir = os.getcwd() + "/models/DIP/"
file_param = "DIP_UNet_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1_last.pt"
params_loaded = torch.load(param_dir + file_param)
unet.load_state_dict(params_loaded)
unet.eval()

# Init adversarial noise setup
from find_adversarial import PAdam_DIP_x
from functools import partial
from operators import proj_l2_ball
# define loss function
# xhat - reconstructed image, x - target image, adv_noise - adversarial noise
# init xhat = Psi_theta(z_tilde)
#loss_adv = lambda adv_noise, xhat, x, meas_op, beta: ( meas_op(xhat) - (meas_op(x) + adv_noise) ).pow(2).sum() - beta * (xhat - x).pow(2).sum() 
from dip_utils import loss_adv
loss_adv_partial = partial(loss_adv,  x = tar_complex, meas_op = OpA, beta = 1e-3)
# init input optimizer (PGD or alternative methods like PAdam)
adv_init_fac = 3
noise_rel = 1e-3
adv_noise_mag = adv_init_fac * noise_rel * measurement.norm(p=2) / np.sqrt(np.prod(measurement.shape[-2:]))
adv_noise_init = adv_noise_mag * torch.randn_like(measurement).to(device)
adv_noise_init.requires_grad = True

# ------------- Projection setup -----------------------------
# radius is the upper bound of the lp-norm of the projeciton operator
radius = noise_rel * measurement.norm(p=2)
# centre is here the centre of the perturbations of the measurements
# since we have chosen to optimize the adv. noise and not adv. measurements (/example)
# centre set to zero freq since measurements are zero-shifted
centre = torch.zeros_like(measurement).to(device)
projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)

# perform PAdam - uses the ADAM optimizer instead of GD and excludes the backtracking line search
adversarial_noise = PAdam_DIP_x(
    loss          = loss_adv_partial,
    x0            = unet(z_tilde), 
    t_in          = adv_noise_init,
    projs         = [projection_l2],
    niter         = 10000,
    stepsize      = 1e-4,
    silent        = False,
)

perturbed_measurement = measurement + adversarial_noise
torch.save(adversarial_noise, os.getcwd() + "/adv_attack_dip/adv_noise.pt")