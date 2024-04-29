from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import os, sys, torch, numpy as np
# ----------- personal imports ----------
from networks import DeepDecoder
from operators import (
    to_complex,
    proj_l2_ball,
    Fourier,
    Fourier_matrix as Fourier_m,
    RadialMaskFunc,
    MaskFromFile,
)
import config
# import for adv. attacks
from find_adversarial import (
    PAdam_DIP_x,
    untargeted_attack,
)
from dip_utils import (
    loss_adv_example, 
    loss_adv_noise, 
    loss_adv_example_white_box,
)
from functools import partial
from copy import deepcopy

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
    # -------- a=1 Samplig patterns --------------------
    #path = os.path.join(config.SP_PATH, "circle"), 
    #filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
    # -------- a=2 Samplig patterns --------------------
    path = config.SP_PATH,
    filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png",
)
mask = mask_fromfile.mask[None]
# Fourier matrix
OpA = Fourier(mask)
OpA_m = Fourier_m(mask)

# set device for operators
OpA_m.to(device)

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

# set device for network
if deep_decoder.device == torch.device("cpu"):
    deep_decoder = deep_decoder.to(device)
assert gpu_avail and deep_decoder.device == device, "for some reason DeepDecoder is on %s even though gpu avail %s"%(deep_decoder.device, gpu_avail)

# JFA's local dir
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val   = os.path.join(config.DATA_PATH, "val")

# same as DIP
sample_idx = 21
tar = torch.load(os.path.join(dir_val, "sample_%.5i_text.pt"%sample_idx) ).to(device)
tar_complex = to_complex(tar[None, None]).to(device)
measurement = OpA(tar_complex).to(device)

# temp dir with network weights and z_tilde
#param_dir = os.path.join(config.SCRATCH_PATH, "DeepDecoder")
param_dir = os.path.join(config.RESULTS_PATH_KADINGIR,"DeepDecoder", "%s_sr%.2f_a2"%(sp_type, sampling_rate))
# Deep decoder input
# load ddinput
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}_k{dim_channels}_nc{num_channels}_{architecture}{additional}".format(
    lr               = 0.005,
    gamma            = 0.98,
    sampling_pattern = "%s_sr%.2f"%(sp_type, sampling_rate),
    dim_channels     = dim_channels,
    num_channels     = num_channels,
    architecture     = archkey,
    #additional       = "",                                  # a=1 multilevel sampling patterns
    additional       = "_a2",                                # a=2 multilevel sampling patterns
)
ddinput = torch.load(os.path.join(param_dir, "ddinput_%s.pt"%(fn_suffix)) )
ddinput = ddinput.to(device)

# load model weights
#param_dir = os.path.join(config.RESULTS_PATH, "DeepDecoder")

file_param = "DeepDecoder_nojit_%s_last.pt"%(fn_suffix)
params_loaded = torch.load( os.path.join(param_dir,file_param) )
deep_decoder.load_state_dict(params_loaded)
deep_decoder.eval()
for p in deep_decoder.parameters():
    p.requires_grad = False

xhat0 = deep_decoder.forward(ddinput)

from torchvision.utils import save_image
plotdir = os.path.join(config.PLOT_PATH, "adversarial_plots", "DeepDecoder", "test_adv_rec")
save_image(xhat0.norm(p=2, dim=(0,1)).detach().cpu(), os.path.join(plotdir, "xhat0_DIP_adv_attack.png") )

# ------------ define adv. attack loss function -----------------------------------------------------------
# xhat - reconstructed image, x - target image, adv_noise - adversarial noise
# init xhat = Psi_theta(z_tilde)
#loss_adv_noise = lambda adv_noise, xhat, x, meas_op, beta: ( meas_op(xhat) - (meas_op(x) + adv_noise) ).pow(2).sum() - beta * (xhat - x).pow(2).sum() 
#TODO: include white box version and confirm that adversarial noise is found in C^N, i.e. for the image not the measurement
#loss_adv_partial = partial(loss_adv_noise,  x = tar_complex, meas_op = OpA, beta = 1e-3)

# measurement or image adversarial attack
adversarial_measurement = True
# init setup for adversarial measurement
if adversarial_measurement:
    loss_adv_partial = partial(loss_adv_example,  meas_op = OpA, beta = 1e-3)
    # init input optimizer for adversarial examples/measurements (PGD or alternative methods like PAdam)
    adv_init_fac     = 3
    noise_rel        = 6e-2
    adv_noise_mag    = adv_init_fac * noise_rel * measurement.norm(p=2) / np.sqrt(np.prod(measurement.shape[-2:]))
    adv_noise_init   = adv_noise_mag * torch.randn_like(measurement).to(device)
    adv_example_init = measurement.clone() + adv_noise_init.clone()
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
    func      = partial(rec_func_adv_noise, xhat=xhat0), # rec,
    t_in_adv  = adv_example_init,                        # yadv[idx_batch : idx_batch + batch_size, ...].clone().requires_grad_(True),
    t_in_ref  = deepcopy(measurement.clone()),           # y0_batch,
    t_out_ref = tar_complex,                           # x0_batch,
    **adv_param
).detach()
torch.save( measurement, os.getcwd() + "/adv_attack_dd/adv_example_noiserel%.2f_%s"%(noise_rel, file_param))

"""
# perform PAdam - uses the ADAM optimizer instead of GD and excludes the backtracking line search
save_adv_noise = False
adversarial_noise_image = PAdam_DIP_x(
    loss          = loss_adv_partial,
    xhat0         = deep_decoder(ddinput), 
    t_in          = adv_noise_init,
    projs         = [projection_l2],
    niter         = 1000,
    stepsize      = 1e-4,
    silent        = False,
)
#adversarial_noise_image = torch.load(os.getcwd() + "/adv_attack_dd/adv_noise_image_dd.pt")
adversarial_noise = OpA(adversarial_noise_image)
perturbed_measurement = measurement + adversarial_noise
save_adv_noise = True
if save_adv_noise:
    torch.save(adversarial_noise_image, os.getcwd() + "/adv_attack_dd/adv_noise_image_dd_%s.pt"%(fn_suffix))
    torch.save(adversarial_noise, os.getcwd()       + "/adv_attack_dd/adv_noise_dd_%s.pt"%(fn_suffix))


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
save_dir = os.path.join(config.RESULTS_PATH, "..", "plots", "adversarial_plots", "DeepDecoder")
fig.savefig( os.path.join(save_dir, "dd_orig_adv_noise_adv_rec_%s.png"%(fn_suffix)) )
"""
