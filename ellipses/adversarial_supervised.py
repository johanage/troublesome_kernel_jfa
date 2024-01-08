from matplotlib import pyplot as plt
import os, sys, torch
from networks import UNet
from operators import (
    Fourier,
    Fourier_matrix as Fourier_m,
    LearnableInverterFourier,
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
inverter = LearnableInverterFourier(config.n, mask_fromfile.mask[None], learnable=False)

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
unet = unet(**unet_params)
param_dir_phase1 = os.getcwd() + "/models/circ_sr0.25/Fourier_UNet_no_jitter_brain_fastmri_256train_phase_1/"
param_dir_phase2 = os.getcwd() + "/models/circ_sr0.25/Fourier_UNet_no_jitter_brain_fastmri_256train_phase_2/"

# get train and validation data
dir_train = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/train/"
dir_val = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/val/"

# same as DIP
v_tar = torch.load(dir_val + "sample_00000.pt").to(device)
v_tar_complex = to_complex(v_tar[None]).to(device)
measurement = OpA(v_tar_complex).to(device)[None]

# load model weights
file_param = "model_weights.pt"
params_loaded = torch.load(param_dir_phase2 + file_param)
unet.load_state_dict(params_loaded)
unet.eval()
from find_adversarial import PGD, PAdam
from functools import partial
from operators import proj_l2_ball
# define loss function
# x - target image, y - measurements, net - DL model, adv_noise - adversarial noise
loss_adv = lambda adv_noise,x,y,net: (net(y + adv_noise) - x).pow(2).pow(.5).sum()
loss_adv_partial = partial(loss_adv, x = v_tar, y = measurement, net = unet)

# init input optimizer (PGD or alternative methods like PAdam)
adv_noise_mag = 0.05
adv_noise_init = adv_noise_mag * torch.rand_like(measurement).to(device)
adv_noise_init.requires_grad = True

# ------------- Projection setup -----------------------------
# radius is the upper bound of the lp-norm of the projeciton operator
radius = torch.tensor(1e-1).to(device)
# centre is here the centre of the measurements - in general the centre of the projection ball 
# centre set to zero freq since measurements are zero-shifted
centre = torch.zeros_like(measurement).to(device)
projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)

# perform PGD
"""
adversarial_noise = PGD(
    loss        = loss_adv_partial,
    t_in        = adv_noise_init,
    projs       = [projection_l2],
    iter        = 50,
    stepsize    = 1e-4,
    maxls       = 50,
    ls_fac      = 0.1,
    ls_severity = 1.0,
    silent      = False,
)
"""
# perform PAdam - uses the ADAM optimizer instead of GD and excludes the backtracking line search
adversarial_noise = PAdam(
    loss        = loss_adv_partial,
    t_in        = adv_noise_init,
    projs       = [projection_l2],
    iter        = 10,
    stepsize    = 1e-4,
    silent      = False,
)


perturbed_measurements = measurement + adversarial_noise
perturbed_targets = unet.forward(perturbed_measurements)
perturbed_images  = (perturbed_targets[:,0]**2 + perturbed_targets[:,1]**2)**.5
fig, axs = plt.subplots( len(perturbed_images), 3, figsize=(10, 10) )
for i, img in enumerate(perturbed_images):
    if len(axs.shape) > 1:
        ax = axs[i]
    else: ax = axs
    ax[0].imshow(v_tar.detach().cpu(), cmap = "Greys_r")#; ax[0].set_title("Original")
    ax[1].imshow(img.detach().cpu(),   cmap = "Greys_r")#; ax[1].set_title("Perturbed")
    ax[2].imshow(torch.abs( v_tar.detach().cpu() - img.detach().cpu() ), cmap = "Greys_r")#; ax[2].set_title("Residuals")

[ax.set_axis_off() for ax in axs.flatten()]
# remove whitespace and set tight layout
fig.tight_layout()
fig.savefig(os.getcwd() + "/adversarial_example.png", bbox_inches = "tight")
