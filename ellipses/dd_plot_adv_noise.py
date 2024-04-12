from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import os, sys, torch, numpy as np
from networks import DeepDecoder 
from operators import (
    Fourier,
    Fourier_matrix as Fourier_m,
    RadialMaskFunc,
    MaskFromFile,
)
import config
from operators import to_complex

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
OpA_m.cpu()

# ----- Network configuration -----
num_channels = 5
dim_channels = 128
archdict = {
    "unetinsp1" : [128, 64, 32, 64, 128],
    "unetinsp2" : [128, 128, 64, 64, 128],
    "flat"      : [dim_channels]*num_channels,
}
archkey = "flat"
deep_decoder_params = {
    "output_channels" : 2,                # 1 : grayscale, 2 : complex grayscale, 3 : RGB
    "channels_up"     : archdict[archkey],
    "out_sigmoid"     : True,
    "act_funcs"       : ["leakyrelu"]*num_channels,
    "kernel_size"     : 1,                 # when using kernel size one we are not using convolution
    "padding_mode"    : "reflect",
    "upsample_sf"     : 2,                # upsample scale factor
    "upsample_mode"   : "bilinear",
    "upsample_first"  : True,
}
deep_decoder = DeepDecoder(**deep_decoder_params)
print("The number of paramters in the DeepDecoder is :", sum(p.numel() for p in deep_decoder.parameters() if p.requires_grad) )
print("The input is of size ", 256**2*2)

# training directory
dir_train = os.path.join(config.DATA_PATH, "train")
# validation dir
dir_val = os.path.join(config.DATA_PATH, "val")
# same as DeepDecoder
tar = torch.load(os.path.join(dir_train,"sample_00000.pt")).cpu()
tar_complex = to_complex(tar[None])
measurement = OpA(tar_complex)[None]

# Deep decoder input
# load ddinput
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}_k{dim_channels}_nc{num_channels}_{architecture}".format(
   lr                = 0.005,
    gamma            = 0.98,
    sampling_pattern = "%s_sr%.2f"%(sp_type, sampling_rate),
    dim_channels     = dim_channels,
    num_channels     = num_channels,
    architecture     = archkey,
)
ddinput = torch.load(os.path.join(config.SCRATCH_PATH, "DeepDecoder", "ddinput_%s.pt"%(fn_suffix)) )
ddinput = ddinput.cpu()

# ------------------------------------------------------------------------------------------
# load model weights normal reconstruction
# ------------------------------------------------------------------------------------------
param_dir = os.path.join(config.SCRATCH_PATH,"DeepDecoder")
file_param = "DeepDecoder_nojit_%s_last.pt"%(fn_suffix)
print("loading OG DeepDecoder net weights from file\n %s in dir\n %s"%(file_param, param_dir))
params_loaded = torch.load( os.path.join(param_dir,file_param) )
deep_decoder.load_state_dict(params_loaded)
# freeze params
for p in deep_decoder.parameters():
    p.requires_grad = False
# disable training layers
deep_decoder.eval()
# reconstruct measurement
img_rec_orig_complex = deep_decoder.forward(ddinput).cpu() 
img_rec_orig = img_rec_orig_complex[0].norm(p=2, dim=0)
# rel. l2-error
rec_orig_rel_l2err = (tar_complex.cpu() - img_rec_orig_complex[0]).norm(p=2)/tar_complex.norm(p=2)
print("---------------------------------------------------------")
print("DeepDecoder reconstruction rel. l2-error", rec_orig_rel_l2err.item())
print("---------------------------------------------------------")
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# load model weights adversarial net
# ------------------------------------------------------------------------------------------
file_param = "DeepDecoder_adv_%s_last.pt"%fn_suffix
print("loading Adv. DeepDecoder net weights from file\n %s in dir\n %s"%(file_param, param_dir))
params_loaded = torch.load( os.path.join(param_dir,file_param) )
deep_decoder.load_state_dict(params_loaded)
# freeze params
for p in deep_decoder.parameters():
    p.requires_grad = False
# disable training layers
deep_decoder.eval()
# reconstruct measurement
img_rec_complex = deep_decoder.forward(ddinput).cpu() 
img_rec = img_rec_complex[0].norm(p=2, dim=0)
# rel. l2-error
rec_rel_l2err = (tar_complex.cpu() - img_rec_complex[0]).norm(p=2)/tar_complex.norm(p=2)
print("---------------------------------------------------------")
print("DeepDecoder reconstruction rel. l2-error", rec_rel_l2err.item())
print("---------------------------------------------------------")
# ------------------------------------------------------------------------------------------

# load noise
#adversarial_noise_image = torch.load(os.getcwd() + "/adv_attack_dd/adv_noise_image_dd_%s.pt"%fn_suffix)
adversarial_noise_image = torch.load(os.path.join(config.RESULTS_PATH, "..","adv_attack_dd", "adv_noise_image_dd_%s.pt"%fn_suffix) )
adversarial_noise = OpA(adversarial_noise_image).cpu()
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
axs[1,1].set_title("Orig. DeepDecoder rec.")
plot_dip_rec            = axs[1,2].imshow(img_rec, cmap=cmap)
axs[1,2].set_title("Adv. DeepDecoder rec.")
plot_rec_adv_noise_orth = axs[1,3].imshow(ehat_adv_orth[0].norm(p=2, dim=0), cmap=cmap)
axs[1,3].set_title("Adv. noise orth. DeepDecoder rec.")
plot_dip_adv_noise_null = axs[1,4].imshow(img_rec_adv_noise_null, cmap=cmap)
axs[1,4].set_title("Adv. noise null DeepDecoder rec.")
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
fn_savefig = os.path.join(config.RESULTS_PATH, "..", "plots/adversarial_plots/DeepDecoder/dd_compare_xadv.png")
#fig.tight_layout()
fig.savefig(fn_savefig)
