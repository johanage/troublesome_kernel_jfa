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
sampling_rate = 0.25
print("sampling rate used is :", sampling_rate)
sp_type = "circle"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, "circle"),
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
)
mask = mask_fromfile.mask[None]
# Fourier matrix
OpA = Fourier(mask)
OpA_m = Fourier_m(mask)
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
    "upsampling"    : "nearest",
}
unet = UNet
unet = unet(**unet_params)
method = "supervised"
dataset = "ellipses"
sampling_pattern_dir = "%s_sr%.2f"%(sp_type, sampling_rate)
method_config = "Fourier_UNet_no_jitter_ellipses_256"
param_dir = os.path.join(config.SCRATCH_PATH, method, dataset, sampling_pattern_dir, method_config, "train_phase_1")
print("The network weights are from : ", param_dir)
# get train and validation data
data_dir = config.TOY_DATA_PATH
dir_train = os.path.join(data_dir, "train")
dir_val = os.path.join(data_dir, "val")

# load final net
file_param = "model_weights.pt"
params_loaded = torch.load( os.path.join(param_dir, file_param) )
unet.load_state_dict(params_loaded)

# define the two images
x      = torch.load( os.path.join(dir_val, "sample_0.pt") )[None, None].to(device)
eta    = 0.07 * x.norm(p=2)
noise  = torch.randn_like(x)
# construct x' s.t. || x - x' || > 2*eta
etaplus = 1e-3 * x.norm(p=2)
xprime = x + (eta + etaplus) * noise / noise.norm(p=2)

# compute measurements
y = OpA(x).to(device)
yprime = OpA(xprime).to(device)
# confirm that the measurement error || Ax - Ax'|| <= eta
assert (y - yprime).norm(p=2) <= eta 

#reconstruct each image
rec      = unet.forward(y)
recprime = unet.forward(yprime) 

# confirm that the reconstruction errors ||Psi(Ax) - x || < eta and || Psi(Ax') - x'|| < eta
breakpoint()
assert (rec      -      x).norm(p=2) < eta
assert (recprime - xprime).norm(p=2) < eta


