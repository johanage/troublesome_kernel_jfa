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

# sampling pattern config
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
    "upsampling"    : "nearest",
}
unet = UNet
# ------ construct network and train -----
unet = unet(**unet_params)
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# JFA's local dir
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val   = os.path.join(config.DATA_PATH, "val")
# temp dir with network weights and z_tilde
param_dir = os.path.join(config.SCRATCH_PATH, "DIP")

# same as DIP
tar = torch.load(os.path.join(dir_train,"sample_00000.pt")).to(device)
tar_complex = to_complex(tar[None]).to(device)
measurement = OpA(tar_complex).to(device)[None]

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
noise_mag = .1
# same shape as SR problem in Ulyanov et al 2018
#random init : z_tilde = noise_mag * torch.rand((unet_params["in_channels"],) + tar.shape)
z_tilde = torch.load(os.path.join(param_dir, "z_tilde_%s_%.2f.pt"%(sp_type, sampling_rate)) )
z_tilde = z_tilde.to(device)

# load model weights
#param_dir = os.path.join(config.RESULTS_PATH, "DIP")
file_param = "DIP_UNet_nojit_lr_0.0005_gamma_0.96_sp_%s_sr%.2f_last.pt"%(sp_type, sampling_rate)
params_loaded = torch.load( os.path.join(param_dir,file_param) )
unet.load_state_dict(params_loaded)
unet.eval()
# Init adversarial noise setup
from find_adversarial import PAdam_DIP_x
from functools import partial
from operators import proj_l2_ball
# define loss function
# xhat - reconstructed image, x - target image, adv_noise - adversarial noise
# init xhat = Psi_theta(z_tilde)
#loss_adv = lambda adv_noise, xhat, x, meas_op, beta: ( meas_op(xhat) - (meas_op(x) + adv_noise) ).pow(2).sum() - beta * (xhat - x).pow(2).sum() 
from dip_utils import loss_adv_noise, loss_adv_example_white_box
#TODO: include white box version and confirm that adversarial noise is found in C^N, i.e. for the image not the measurement
#loss_adv_partial = partial(loss_adv_noise,  x = tar_complex, meas_op = OpA, beta = 1e-3)
loss_adv_partial = partial(loss_adv_example_white_box,  x = tar_complex, meas_op = OpA, beta = 1e-3)

# measurement or image adversarial attack
adversarial_measurement = False
# init setup for adversarial measurement
if adversarial_measurement:
    # init input optimizer for adversarial examples/measurements (PGD or alternative methods like PAdam)
    adv_init_fac   = 3
    noise_rel      = 1e-2
    adv_noise_mag  = adv_init_fac * noise_rel * measurement.norm(p=2) / np.sqrt(np.prod(measurement.shape[-2:]))
    adv_noise_init = adv_noise_mag * torch.randn_like(measurement).to(device)
    adv_noise_init.requires_grad = True

    # ------------- Projection setup -----------------------------
    # radius is the upper bound of the lp-norm of the projeciton operator
    radius = noise_rel * measurement.norm(p=2)
    # centre is here the centre of the perturbations of the measurements
    # since we have chosen to optimize the adv. noise and not adv. measurements (/example)
    # centre set to zero freq since measurements are zero-shifted
    centre = torch.zeros_like(measurement).to(device)
    projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)

# init setup for adversarial image
else:
    # init noise for adv image noise
    adv_init_fac   = 3
    noise_rel      = 1e-1
    adv_noise_mag  = adv_init_fac * noise_rel
    target_image   = tar_complex[None]
    adv_noise_init = adv_noise_mag * torch.randn_like(target_image).to(device)
    adv_noise_init.requires_grad = True
    # ------------- Projection setup -----------------------------
    # radius is the upper bound of the lp-norm of the projeciton operator
    radius = noise_rel * target_image.norm(p=2)
    centre = torch.zeros_like(target_image).to(device)
    projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)

# perform PAdam - uses the ADAM optimizer instead of GD and excludes the backtracking line search
save_adv_noise = False
#"""
adversarial_noise_image = PAdam_DIP_x(
    loss          = loss_adv_partial,
    xhat0         = unet(z_tilde), 
    t_in          = adv_noise_init,
    projs         = [projection_l2],
    niter         = 1000,
    stepsize      = 1e-4,
    silent        = False,
)
save_adv_noise = True
#"""
#adversarial_noise_image = torch.load(os.getcwd() + "/adv_attack_dip/adv_noise_image_dip_x.pt")
adversarial_noise = OpA(adversarial_noise_image)
perturbed_measurement = measurement + adversarial_noise
if save_adv_noise:
    torch.save(adversarial_noise_image, os.getcwd() + "/adv_attack_dip/adv_noise_image_dip_x_%s_sr%.2f.pt"%(sp_type, sampling_rate))
    torch.save(adversarial_noise, os.getcwd()       + "/adv_attack_dip/adv_noise_dip_x_%s_sr%.2f.pt"%(sp_type, sampling_rate))


# plot the following
#  - adversarial noise and example
#  - adjoints of meas and perturbed meas
#  - gt image
fig, axs = plt.subplots(1, 5, figsize = (15, 5))
cmap = "Greys_r"
[ax.set_axis_off() for ax in axs]
plot_gt_image      = axs[0].imshow(tar.cpu(), cmap=cmap)
axs[0].set_title("Original image")
img_adj_meas       = OpA.adj(measurement).detach().cpu()
plot_adjmeas_image = axs[1].imshow(img_adj_meas[0].norm(p=2, dim=0), cmap=cmap)
axs[1].set_title("Adjoint of measurements")
# adversarial noise image
adv_noise_img = adversarial_noise_image.detach().cpu()[0].norm(p=2, dim=0)
plot_adv_noise  = axs[2].imshow(adv_noise_img, cmap=cmap)
axs[2].set_title("Adversarial noise (image)")
adj_adv_example = OpA.adj(perturbed_measurement).detach().cpu()
plot_adv_example_pert_meas = axs[3].imshow(adj_adv_example[0].norm(p=2, dim=0), cmap=cmap)
axs[3].set_title("Adjoint of perturbed measurements")
plot_adv_example = axs[4].imshow(tar.cpu() + adv_noise_img, cmap=cmap)
axs[4].set_title("Adversarial example (image)")
for ax,plot in zip(axs,[plot_gt_image, plot_adjmeas_image,plot_adv_noise,plot_adv_example_pert_meas, plot_adv_example]):
    divider = mal(ax)
    cax     = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(plot, cax=cax)
# save figure
save_dir = os.path.join(config.RESULTS_PATH, "..", "plots", "adversarial_plots", "DIP")
fig.savefig(os.path.join(save_dir, "dip_x_orig_adv_noise_adv_rec_%s_sr%.2f.png"%(sp_type, sampling_rate)))
