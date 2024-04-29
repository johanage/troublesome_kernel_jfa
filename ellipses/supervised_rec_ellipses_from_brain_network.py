from matplotlib import pyplot as plt
import os, sys, torch, numpy as np
from torchvision.utils import save_image
from copy import deepcopy
# --- Local imports ------------
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
#OpA.to(device)
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
    "upsampling"   : "nearest",
}
unet = UNet(**unet_params)
#param_dir = os.path.join(config.RESULTS_PATH, "supervised/circ_sr0.25/Fourier_UNet_jitter_brain_fastmri_256eta_100.000_train_phase_2/")
param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "supervised", "circle_sr0.25_a2", "Fourier_UNet_no_jitter_brain_fastmri_256", "train_phase_1")
jitter_level = 0
#param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "supervised", "circle_sr0.25_a2", "Fourier_UNet_jitter_brain_fastmri_256", "eta_%.3f_train_phase_1"%jitter_level)
# load model weights
file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(param_dir, file_param) )
unet.load_state_dict(params_loaded)
unet.to(device)

# get train and validation data
dir_val_fastmri  = os.path.join(config.DATA_PATH, "val")
dir_val_ellipses = os.path.join(config.TOY_DATA_PATH, "val")

# set sample id for the sample with a textual detail 
sample_idx = 21
# read sample, real -> complex and compute measurement
v_tar_fastmri = torch.load(os.path.join(dir_val_fastmri, "sample_%.5i_text.pt"%sample_idx) ).to(device)
v_tar_fastmri_complex = to_complex(v_tar_fastmri[None, None]).to(device)
measurement_fastmri = OpA(v_tar_fastmri_complex).to(device)

# read sample, real -> complex and compute measurement
v_tar_ellipses = torch.load(os.path.join(dir_val_ellipses, "sample_%i_text.pt"%sample_idx) ).to(device)
v_tar_ellipses_complex = to_complex(v_tar_ellipses[None, None]).to(device)
measurement_ellipses = OpA(v_tar_ellipses_complex).to(device)

# compute reconstructions
rec_fastmri  = unet.forward(measurement_fastmri)
rec_ellipses = unet.forward(measurement_ellipses)
# from complex to real
img_rec_fastmri  = rec_fastmri.norm(p=2, dim=1)[0]
img_rec_ellipses = rec_ellipses.norm(p=2, dim=1)[0]

# ------------------ Plot adversarial example ------------------
plotdir = os.path.join(config.PLOT_PATH, "distr_shift", "supervised", "eta_%.1f"%jitter_level)
# save reconstructions
save_image(img_rec_fastmri.detach().cpu(), os.path.join(plotdir, "rec_fastmri_sample_%i.pdf"%sample_idx) )
save_image(img_rec_ellipses.detach().cpu(), os.path.join(plotdir, "rec_ellipses_sample_%i.pdf"%sample_idx) )
# save originals
save_image(v_tar_fastmri, os.path.join(plotdir, "original_image_fastmri_sample_%i.pdf"%sample_idx) )
save_image(v_tar_ellipses, os.path.join(plotdir, "original_image_ellipses_sample_%i.pdf"%sample_idx) )
