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
# --------- measurment configuration ----------------
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[-1]
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, "circle"),
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
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
param_dir = os.path.join(config.SCRATCH_PATH, "DIP")
# loading the same z_tilde as used in the original and adversarial reconstruction process
z_tilde = torch.load(os.path.join(param_dir, "z_tilde_%s_%.2f.pt"%(sp_type, sampling_rate)))
z_tilde = z_tilde.to(device)

# ------------------------------------------------------------------------------------------
# load model weights normal reconstruction
# ------------------------------------------------------------------------------------------
file_param = "DIP_UNet_nojit_lr_0.0005_gamma_0.96_sp_%s_sr%.2f_last.pt"%(sp_type, sampling_rate)
params_loaded = torch.load( os.path.join(param_dir,file_param) )
unet.load_state_dict(params_loaded)
# freeze params
for p in unet.parameters():
    p.requires_grad = False
# disable training layers
unet.eval()
# reconstruct measurement
img_rec_orig_complex = unet.forward(z_tilde).cpu() 
img_rec_orig = img_rec_orig_complex[0].norm(p=2, dim=0)
# rel. l2-error
rec_orig_rel_l2err = (tar_complex.cpu() - img_rec_orig_complex[0]).norm(p=2)/tar_complex.norm(p=2)
print("DIP_x reconstruction rel. l2-error", rec_orig_rel_l2err)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# load model weights adversarial net
# ------------------------------------------------------------------------------------------
param_dir = os.path.join(config.RESULTS_PATH, "DIP", "adv_attack")
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
# ------------------------------------------------------------------------------------------

# load noise
adversarial_noise_image = torch.load(os.getcwd() + "/adv_attack_dip/adv_noise_image_dip_x.pt")
adversarial_noise = OpA(adversarial_noise_image)
perturbed_measurement = measurement + adversarial_noise

#TODO: fix reconstruction image of the original image and the adversarial reconstruction!!! 
# plot comparison 
fig, axs = plt.subplots(2, 5, figsize = (25, 5))
cmap = "Greys_r"
[ax.set_axis_off() for ax in axs.flatten()]

img_adv_noise = adversarial_noise_image.detach().cpu()
# orthogonal component of adversarial image noise aka A^*delta_adv
img_adv_noise_orth = OpA.adj(OpA(img_adv_noise))
# nullspace component of adversarial image noise aka A^*delta_adv
img_adv_noise_null = img_adv_noise - img_adv_noise_orth

# orthogonal component of image aka x^\perp = A^*Ax
img_orth = OpA.adj(OpA(tar_complex))
# orthoghonal component of reconstruction aka xhat^perp = A^*A(xhat)
img_rec_orth = OpA.adj(OpA(img_rec_complex.cpu()))

# null space component of image aka x_det = x - x^\perp
img_null  = tar_complex - img_orth
ehat_adv_orth = OpA.adj(OpA(img_rec_complex.cpu() - img_rec_orig_complex.cpu()))
img_rec_adv_noise_null_complex = img_rec_complex.cpu() - img_rec_orig_complex.cpu() - ehat_adv_orth
img_rec_adv_noise_null = img_rec_adv_noise_null_complex[0].norm(p=2, dim=0)
# adversarial noise/example plots wihtout reconstruction
plot_adv_noise          = axs[0,0].imshow(img_adv_noise[0].norm(p=2, dim=0), cmap=cmap)
axs[0,0].set_title("Adv. noise")
plot_adv_noise_orth     = axs[0,1].imshow(img_adv_noise_orth[0].norm(p=2, dim=0), cmap=cmap)
axs[0,1].set_title("Adv. noise orth")
plot_adv_noise_null     = axs[0,2].imshow(img_adv_noise_null[0].norm(p=2, dim=0), cmap=cmap)
axs[0,2].set_title("Adv. noise null")
plot_adv_example        = axs[0,3].imshow((tar_complex.cpu() + img_adv_noise)[0].norm(p=2, dim=0), cmap=cmap)
axs[0,3].set_title("Adv. example")
plot_gt_image           = axs[0,4].imshow(tar.cpu(), cmap=cmap)
axs[0,4].set_title("GT")
# gt image and adversarial reconstruction plots
plot_gt_image           = axs[1,0].imshow(tar.cpu(), cmap=cmap)
axs[1,0].set_title("GT")
plot_rec_image          = axs[1,1].imshow(img_rec_orig.cpu(), cmap=cmap)
axs[1,1].set_title("Orig. DIP rec.")
plot_dip_rec            = axs[1,2].imshow(img_rec, cmap=cmap)
axs[1,2].set_title("Adv. DIP rec.")
plot_rec_adv_noise_orth = axs[1,3].imshow(ehat_adv_orth[0].norm(p=2, dim=0), cmap=cmap)
axs[1,3].set_title("Adv. noise orth. DIP rec.")
plot_dip_adv_noise_null = axs[1,4].imshow(img_rec_adv_noise_null, cmap=cmap)
axs[1,4].set_title("Adv. noise null DIP rec.")
for ax,plot in zip(axs.flatten(),[
    plot_adv_noise, 
    plot_adv_noise_orth, 
    plot_adv_noise_null, 
    plot_adv_example, 
    plot_gt_image, 
    plot_gt_image, 
    plot_rec_image, # original reconstruction
    plot_dip_rec,   # adversarial reconsruction
    plot_rec_adv_noise_orth, 
    plot_dip_adv_noise_null
]):
    divider = mal(ax)
    cax     = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(plot, cax=cax)
# save figure
fn_savefig = os.path.join(config.RESULTS_PATH, "..", "plots/adversarial_plots/DIP/dip_x_compare_xadv.png")
#fig.tight_layout()
fig.savefig(fn_savefig)
