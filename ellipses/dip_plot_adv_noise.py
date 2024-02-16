from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
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
    "upsampling"    : "nearest",#"trans_conv",
}
unet = UNet
# ------ construct network and train -----
unet = unet(**unet_params)
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# training directory
dir_train = os.path.join(config.DATA_PATH, "train")
# same as DIP
tar = torch.load(os.path.join(dir_train,"sample_00000.pt")).to(device)
tar_complex = to_complex(tar[None]).to(device)
measurement = OpA(tar_complex).to(device)[None]

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
noise_mag = .1
# same shape as SR problem in Ulyanov et al 2018
#random init : z_tilde = noise_mag * torch.rand((unet_params["in_channels"],) + tar.shape)
z_tilde = torch.load(os.getcwd() + "/adv_attack_dip/z_tilde.pt")
z_tilde = z_tilde.to(device)

# load model weights
param_dir = os.path.join(config.RESULTS_PATH, "DIP/adv_attack")
file_param = "DIP_UNet_adv_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1_last.pt"
params_loaded = torch.load( os.path.join(param_dir,file_param) )
unet.load_state_dict(params_loaded)
# freeze params
for p in unet.parameters():
    p.requires_grad = False
# disable training layers
unet.eval()
# reconstruct measurement
img_rec_complex = unet.forward(z_tilde).cpu() 
img_rec = img_rec_complex[0].norm(p=2, dim=0)
# rel. l2-error
rec_rel_l2err = (tar_complex.cpu() - img_rec_complex[0]).norm(p=2)/tar_complex.norm(p=2)
print("DIP_x reconstruction rel. l2-error", rec_rel_l2err)
# load noise
adversarial_noise = torch.load(os.getcwd() + "/adv_attack_dip/adv_noise_dip_x.pt")
perturbed_measurement = measurement + adversarial_noise

# plot comparison 
fig, axs = plt.subplots(1, 6, figsize = (15, 5))
cmap = "Greys_r"
[ax.set_axis_off() for ax in axs]
plot_gt_image    = axs[0].imshow(tar.cpu(), cmap=cmap)
# orthogonal component of adversarial image noise aka A^*delta_adv
img_adv_noise    = OpA.adj(adversarial_noise).detach().cpu()
# orthogonal component of image aka x^\perp = A^*Ax
img_orth = OpA.adj(OpA(tar_complex))
# null space component of image aka x_det = x - x^\perp
img_null  = tar_complex - img_orth
ehat_adv_orth = OpA.adj(OpA(img_rec_complex.cpu() - tar_complex.cpu()))
img_adv_noise_null_complex = img_rec_complex.cpu() - tar_complex.cpu() - ehat_adv_orth
img_adv_noise_null         = img_adv_noise_null_complex[0].norm(p=2, dim=0)
# make plots
plot_adv_noise          = axs[1].imshow(img_adv_noise[0].norm(p=2, dim=0), cmap=cmap)
plot_rec_adv_noise_orth = axs[2].imshow(ehat_adv_orth[0].norm(p=2, dim=0), cmap=cmap)
plot_adv_example        = axs[3].imshow(tar.cpu() + img_adv_noise[0].norm(p=2, dim=0), cmap=cmap)
plot_dip_rec            = axs[4].imshow(img_rec, cmap=cmap)
plot_dip_adv_noise_null = axs[5].imshow(img_adv_noise_null, cmap=cmap)
for ax,plot in zip(axs,[
    plot_gt_image, 
    plot_adv_noise, 
    plot_rec_adv_noise_orth, 
    plot_adv_example, 
    plot_dip_rec, 
    plot_dip_adv_noise_null
]):
    divider = mal(ax)
    cax     = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(plot, cax=cax)
# save figure
fn_savefig = os.path.join(config.RESULTS_PATH, "..", "plots/adversarial_plots/DIP/dip_x_compare_xadv.png")
#fig.tight_layout()
fig.savefig(fn_savefig)
