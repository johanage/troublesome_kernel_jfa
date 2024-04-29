# load installed libs
import os
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
# from local scripts
from data_management import (
    IPDataset, 
    ToComplex, 
    AddDetail,
    SimulateMeasurements, 
    Jitter,
)
from networks import UNet
from operators import (
    to_complex,
    Fourier,
    Fourier_matrix as Fourier_m,
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
sampling_rate = 0.25
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
    #"upsampling"    : "trans_conv",
}
# ------ load network  -----
unet = UNet(**unet_params)
# set device
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# load state dict
param_dir = "supervised/circle_sr0.25/Fourier_UNet_no_jitter_brain_fastmri_256/train_phase_1"
file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(config.SCRATCH_PATH,param_dir,file_param))
unet.load_state_dict(params_loaded)

# set network in evaluation mode
unet.eval()
for p in unet.parameters():
    p.requires_grad = False

# set directories
savedir = os.path.join(config.DATA_PATH, "train")
plotdir = os.path.join(config.PLOT_PATH, "false_negative")

# ------- Load the image: x  --------------------------------------------------
idx = "00042"
loadfn = os.path.join(savedir, "sample_{idx}.pt".format(idx=idx))
data = torch.load(loadfn)

# ------ Get random Gaussian vectors --------
gauss_pert = 0.1*torch.randn(size=(100,2,) + data.size())/data.norm(p=2)

# ------ Compute the measurements of the perturbed sample -------
measurement_gauss_pert = [OpA(to_complex(data[None,None] + x[None])).to(device) for x in gauss_pert]

# --- Reconstruct images  -----------------------------------------------
rec_gauss_pert = torch.stack([unet.forward(measurement)[0].norm(p=2, dim=0, keepdim=True) for measurement in measurement_gauss_pert])
# --------- Gridplot reconstructions ------------
from torchvision.utils import make_grid, save_image
grid_imgs = make_grid(rec_gauss_pert, nrow = 10)
save_image(grid_imgs, os.path.join(plotdir, "rec_images.png"))

# --- Plot reconstructions ----------------------------------------------
"""reclist = [data_detailed_image, rec_det_im_trans, rec_det_im, rec_large_det, rec_im]
fig, axs = plt.subplots(1,len(reclist), figsize=(20,4))
cmap = "Greys_r"
plots_rec = [axs[i].imshow(reclist[i].detach().cpu().norm(p=2, dim=(0,1)), cmap=cmap) for i in range(len(reclist))]
for ax,plot in zip(axs,plots_rec):
    divider = mal(ax)
    cax     = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(plot, cax=cax)
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "rec_images.png"), dpi=140)
"""
