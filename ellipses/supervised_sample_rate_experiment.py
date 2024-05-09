"""
Training script for supervised learning using a UNet.
UNet is implemented in networks.py.
-----------------------------------------------------
Terms
    - jitter, means that noise, typically Gaussian, is added to the data while training to reduce overfit
"""
# load installed libs
import os, sys
import matplotlib as mpl
import torch
import torchvision
from torchvision.transforms import v2
# from local scripts
from data_management import IPDataset, SimulateMeasurements, ToComplex, Jitter
from networks import UNet
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    to_complex,
    LearnableInverterFourier,
    RadialMaskFunc,
    MaskFromFile,
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
#sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sr_list = [0.03, 0.05, 0.10, 0.20]
#sampling_rate = sr_list[0]
sampling_rate = sr_list[int(sys.argv[1])]
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, sp_type), # circular pattern
    #path = config.SP_PATH,
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate) # sampling_rate *100 % sr, a = 1, r0 = 2, nlevles = 50 
    #filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png" # circular pattern, 25 % sr, a = 2, r0 = 2, nlevels = 50
)
mask = mask_fromfile.mask[None]

# Fourier matrix
OpA_m = Fourier_m(mask)
# Fourier operator
OpA = Fourier(mask)
# init learnable inversion operator
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
unet = UNet(**unet_params)
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# ------- Load network weights ------------
# /mn/nam-shub-02/scratch/johanfag/supervised/circle_sr0.05/Fourier_UNet_no_jitter_brain_fastmri_256/train_phase_2
param_dir = os.path.join(
    # ------------- pure no jit networks -------------------------------- 
    #config.SCRATCH_PATH,
    #"supervised/%s_sr%.2f/Fourier_UNet_no_jitter_brain_fastmri_256_single_sample"%(sp_type, sampling_rate),
    # ------------- low jittering -> no jit training  --------------------
    config.RESULTS_PATH_KADINGIR,
    "supervised/%s_sr%.2f%s/Fourier_UNet_no_jitter_brain_fastmri_256"%(sp_type, sampling_rate, "_a1"),
    "train_phase_1",
)

file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(param_dir,file_param))
unet.load_state_dict(params_loaded)

# ----- data configuration -----
"""train_data_params = {
    "path": config.DATA_PATH,
}
train_data = IPDataset
train_data = train_data("train", **train_data_params)
# --- Set train_data to single sample ----------
train_data.files = [x for x in train_data.files if "%.5i"%sample_idx in x]
sample = torch.load(train_data.files[0])
"""
# ------ Load sample and reconstruct image -------
sample_idx = 0#21
#sample = torch.load(os.path.join(config.DATA_PATH, "val", "sample_%.5i_text.pt"%sample_idx) )
sample = torch.load(os.path.join(config.DATA_PATH, "train", "sample_%.5i.pt"%sample_idx) )
measurement = OpA(to_complex(sample[None, None])).to(device)
adj_meas = OpA.adj(measurement).cpu()
torchvision.utils.save_image(
    adj_meas.norm(p=2, dim=(0,1)), 
    os.path.join(config.PLOT_PATH, "adjoint_rec", "adjoint_%s_sr%.2f_%s_nlevel50_r0_2.png"%(sp_type, sampling_rate, "_a1") )
)
torchvision.utils.save_image(
    sample,
    os.path.join(config.PLOT_PATH, "adjoint_rec", "sample%.5i.png"%(sample_idx) )
)

# ----- Reconstruct sample ------
unet.eval()
for p in unet.parameters():
    p.requires_grad = False
rec = unet.forward(measurement).cpu()
plotdir = os.path.join(config.PLOT_PATH,"supervised", "sample_rate_experiment") 
torchvision.utils.save_image(rec.norm(p=2, dim=(0,1)), os.path.join(plotdir, "supervised_final_rec_%s_sr%.2f_sample%i.png"%(sp_type, sampling_rate, sample_idx)))
rec_err = (rec.norm(p=2, dim=(0,1)) - sample).abs()
recerr_scale = 1/rec_err.max()
torchvision.utils.save_image(recerr_scale * rec_err, os.path.join(plotdir, "supervised_final_rec_err_%s_sr%.2f_sample%i.png"%(sp_type, sampling_rate, sample_idx)))
