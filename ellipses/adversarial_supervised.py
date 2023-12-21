from matplotlib import pyplot as plt
import os, torch
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

from find_adversarial import PGD
from functools import partial

# define loss function
# x - target image, y - measurements, net - DL model, adv_noise - adversarial noise
loss_adv = lambda x,y,net,adv_noise: (net(y + adv_noise) - x)



out_pgd = PGD(
    loss        = loss_adv,
    t_in        = torch.rand_like(measurement),
    projs       = ,
    iter        : int   = 50,
    stepsize    : float = 1e-2,
    maxls       : int   = 50,
    ls_fac      : float = 0.1,
    ls_severity : float = 1.0,
    silent      : bool  = False,
)
