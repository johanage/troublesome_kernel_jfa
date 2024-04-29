from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import os, sys, torch, numpy as np
# ----- Personal imports -------
from networks import UNet
from operators import (
    to_complex,
    proj_l2_ball, 
    Fourier,
    Fourier_matrix as Fourier_m,
    RadialMaskFunc,
    MaskFromFile,
)
import config
# adversarial attack imports
from functools import partial
from copy import deepcopy
from dip_utils import (
    _reconstructDIP, 
    loss_adv_example, 
    loss_adv_example_white_box,
)
from find_adversarial import (
    untargeted_attack,
    PAdam_DIP_x,
)


#  -------- set device ----------------------
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
    #path = os.path.join(config.SP_PATH, "circle"),
    #filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
    # -------- a=2 Samplig patterns --------------------
    path = config.SP_PATH,
    filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png",
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

# get train and validation data
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val = os.path.join(config.DATA_PATH, "val")

# same as DIP
sample_idx = 21
v_tar = torch.load(os.path.join(dir_val, "sample_%.5i_text.pt"%sample_idx) ).to(device)
v_tar_complex = to_complex(v_tar[None, None]).to(device)
measurement = OpA(v_tar_complex).to(device)

# temp dir with network weights and z_tilde
#param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "DIP")
param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "a2")

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
z_tilde = torch.load(os.path.join(param_dir, "z_tilde_%s_%.2f.pt"%(sp_type, sampling_rate)) )
z_tilde = z_tilde.to(device)

# load model weights
file_param = "DIP_UNet_nojit_lr_0.0001_gamma_0.99_step_100_sp_%s_sr%.2f_a2_last.pt"%(sp_type, sampling_rate)
params_loaded = torch.load( os.path.join(param_dir, file_param) )
unet.load_state_dict(params_loaded)

# --------- Adv. attack loss functions -------------------------------------------------------------------------------------------------------
# xhat - reconstructed image, x - target image, adv_noise - adversarial noise
# init xhat = Psi_theta(z_tilde)
# loss_adv = lambda adv_noise, xhat, x, meas_op, beta: ( meas_op(xhat) - (meas_op(x) + adv_noise) ).pow(2).sum() - beta * (xhat - x).pow(2).sum() 
# --------------------------------------------------------------------------------------------------------------------------------------------
# measurement (black box) or image (white box) adversarial attack
adversarial_measurement = True
#TODO: try implementation of white box with untargeted attack
# init setup for adversarial measurement
if adversarial_measurement:
    # ------- Black box ------------------
    #l_adv = ||A xhat - y_adv||_2^2 - beta * || x - xhat||_2^2
    loss_adv_partial = partial(loss_adv_example,  meas_op = OpA, beta = 1e-3)

    # init input optimizer for adversarial examples/measurements (PGD or alternative methods like PAdam)
    adv_init_fac   = 3
    noise_rel      = 6e-2
    adv_noise_mag  = adv_init_fac * noise_rel * measurement.norm(p=2) / np.sqrt(np.prod(measurement.shape[-2:]))
    adv_noise_init = adv_noise_mag * torch.randn_like(measurement).to(device)
    adv_example_init = measurement + adv_noise_init
    adv_example_init.requires_grad = True
    adv_noise_init.requires_grad = True

    # ------------- Projection setup -----------------------------
    # radius is the upper bound of the lp-norm of the projeciton operator
    radius = noise_rel * measurement.norm(p=2)
    # centre is here the centre of the perturbations of the measurements
    # since we have chosen to optimize the adv. noise and not adv. measurements (/example)
    # centre set to zero freq since measurements are zero-shifted
    #centre = torch.zeros_like(measurement).to(device)
    centre = measurement.clone().to(device)
    projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)

# init setup for adversarial image
else:
    # -------- White box ------------------
    loss_adv_partial = partial(loss_adv_example_white_box,  x = tar_complex, meas_op = OpA, beta = 1e-3)

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
"""
# -------- PAdam_DIP_x -------------------------------------
adversarial_noise_image = PAdam_DIP_x(
    loss          = loss_adv_partial,
    xhat0         = unet(z_tilde), 
    t_in          = adv_noise_init,
    projs         = [projection_l2],
    niter         = 1000,
    stepsize      = 1e-4,
    silent        = False,
)
#"""

# --------- Untargeted attack ---------------------------------'
# set unet in evaluation mode
unet.eval()
for p in unet.parameters():
    p.requires_grad = False

xhat0 = unet.forward(z_tilde)

from torchvision.utils import save_image
plotdir = os.path.join(config.PLOT_PATH, "adversarial_plots", "DIP", "test_adv_rec")
save_image(xhat0.norm(p=2, dim=(0,1)).detach().cpu(), os.path.join(plotdir, "xhat0_DIP_adv_attack.png") )

adv_param = {
    "codomain_dist" : loss_adv_partial,
    "domain_dist"   : None,
    "mixed_dist"    : None,
    "weights"       : (1.0, 1.0, 1.0),
    "optimizer"     : partial(PAdam_DIP_x, xhat0 = xhat0 ),
    "projs"         : [projection_l2],
    "niter"         : 1000,
    "stepsize"      : 1e-3,
}


rec_func_adv_noise = lambda y, xhat : xhat

# compute adversarial example for batch
measurement = untargeted_attack(
    func      = partial(rec_func_adv_noise, xhat=xhat0), #rec,
    t_in_adv  = adv_example_init,  # yadv[idx_batch : idx_batch + batch_size, ...].clone().requires_grad_(True),
    #t_in_ref  = measurement.clone(),    # y0_batch,
    t_in_ref  = deepcopy(measurement.clone()),    # y0_batch,
    t_out_ref = v_tar_complex,  # x0_batch,
    **adv_param
).detach()
torch.save( measurement, os.getcwd() + "/adv_attack_dip/adv_example_noiserel%.2f_%s"%(noise_rel, file_param))

#--------------------------------------------------------------

#adversarial_noise_image = torch.load(os.getcwd() + "/adv_attack_dip/adv_noise_image_dip_x.pt")
#adversarial_noise = OpA(adversarial_noise_image)
#perturbed_measurement = measurement + adversarial_noise
save_adv_noise = False
if save_adv_noise:
    torch.save(adversarial_noise, os.getcwd()       + "/adv_attack_dip/adv_noise_dip_x_%s_sr%.2f.pt"%(sp_type, sampling_rate))


# plot the following
#  - adversarial noise and example
#  - adjoints of meas and perturbed meas
#  - gt image
"""
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
"""
