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
# plot stuff
cmap = "Greys_r"
isave = 0
fn_evolution = "supervised_evolution_circ_sr0.25"
fig_evo, axs_evo = plt.subplots(2,10,figsize=(50,10) )
[ax.set_axis_off() for ax in axs_evo.flatten()]

# run evo
for indices, param_dir in zip([ [0,6,12,15,18,24,30,35], [3, 6]], [param_dir_phase1, param_dir_phase2]):
    for i in indices:
        if i > 0:
            file_param = "model_weights_epoch%i.pt"%i
            params_loaded = torch.load(param_dir + file_param)
            unet.load_state_dict(params_loaded)
        v_pred = unet.forward(measurement)
        # plot evolution
        pred_cpu = v_pred.detach().cpu()
        abs_img = lambda  x : (x[0]**2 + x[1]**2)**.5
        impred = abs_img(pred_cpu[0])
        imres = impred - v_tar.detach().cpu()
        axs_evo[0,isave].imshow(impred, cmap=cmap)
        axs_evo[1,isave].imshow(imres, cmap=cmap)
        isave +=1
fig_evo.tight_layout()
fig_evo.savefig(os.getcwd() + "/" + fn_evolution + ".png", bbox_inches="tight")
