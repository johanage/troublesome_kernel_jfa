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
#param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "supervised", "circle_sr0.25_a2", "Fourier_UNet_no_jitter_brain_fastmri_256", "train_phase_1")
eta = 25
param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "supervised", "circle_sr0.25_a2", "Fourier_UNet_jitter_mod_brain_fastmri_256", "eta_%.3f_train_phase_1"%eta)
#param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "supervised", "circle_sr0.25_a2", "Fourier_UNet_jitter_brain_fastmri_256", "eta_%.3f_train_phase_1"%eta)
# load model weights
file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(param_dir, file_param) )
unet.load_state_dict(params_loaded)
unet.to(device)

# get train and validation data
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val = os.path.join(config.DATA_PATH, "val")

# same as DIP
sample_idx = 21
v_tar = torch.load(os.path.join(dir_val, "sample_%.5i_text.pt"%sample_idx) ).to(device)
v_tar_complex = to_complex(v_tar[None, None]).to(device)
measurement = OpA(v_tar_complex).to(device)
og_meas = deepcopy(measurement.clone())
from find_adversarial import PGD, PAdam, untargeted_attack
from functools import partial
from operators import proj_l2_ball


# define loss function
# x - target image, y - measurements, net - DL model, adv_noise - adversarial noise
#loss_adv = lambda adv_noise,x,y,net: (net(y + adv_noise) - x).norm(p=2,dim=1).max() # l-infinity norm for vector with complex entries
loss_adv = lambda adv_noise,x,y,net: (net(y + adv_noise) - x).pow(2).pow(.5).sum()# / x.shape[-1]
loss_adv_partial = partial(loss_adv, x = v_tar_complex, y = measurement, net = unet)

# init input optimizer (PGD or alternative methods like PAdam)
adv_init_fac = 3
noise_rel = 0.01
adv_noise_mag = adv_init_fac * noise_rel * measurement.norm(p=2) / np.sqrt(np.prod(measurement.shape[-2:])) 

# make init adversarial noise vector
adv_noise_init = adv_noise_mag * torch.randn_like(measurement).to(device)
#adv_noise_init.requires_grad = True

# make init adversarial example vector
adv_example_init = measurement + adv_noise_init
adv_example_init.requires_grad = True

# ------------- Projection setup -----------------------------
# radius is the upper bound of the lp-norm of the projection operator
radius = (8e-2 * measurement.norm(p=2)).to(device)
# ------ centre is here the centre of the measurements - in general the centre of the projection ball ------------- 
# centre set to zero freq since measurements are zero-shifted
#centre = torch.zeros_like(measurement).to(device)
# centre set to zero freq since measurements are zero-shifted
centre = measurement.clone().detach()
projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)

# perform PGD
"""
adversarial_noise = PGD(
    loss        = loss_adv_partial,
    t_in        = adv_noise_init,
    projs       = [projection_l2],
    iter        = 50,
    stepsize    = 1e-4,
    maxls       = 50,
    ls_fac      = 0.1,
    ls_severity = 1.0,
    silent      = False,
)
"""
"""
# perform PAdam - uses the ADAM optimizer instead of GD and excludes the backtracking line search
adversarial_noise = PAdam(
    loss        = loss_adv_partial,
    t_in        = adv_noise_init,
    projs       = [projection_l2],
    niter       = 50,
    stepsize    = 1e-3,
    silent      = False,
)
"""
# loss functions
mseloss = torch.nn.MSELoss(reduction="sum")
def _complexloss(reference, prediction):
    loss = mseloss(reference, prediction)# / reference.norm(p=2)
    return loss

adv_param = {
    "codomain_dist" : _complexloss,
    "domain_dist"   : None,
    "mixed_dist"    : None,
    "weights"       : (1.0, 1.0, 1.0),
    "optimizer"     : PAdam, #adv_optim,
    "projs"         : [projection_l2], #None,
    "niter"         : 1000,
    "stepsize"      : 1e-4,
}

# set unet in evaluation mode
unet.eval()
for p in unet.parameters():
    p.requires_grad = False

# compute adversarial example for batch
measurement = untargeted_attack(
    func      = lambda y: unet.forward(y), #rec,
    t_in_adv  = adv_example_init,  # yadv[idx_batch : idx_batch + batch_size, ...].clone().requires_grad_(True),
    #t_in_ref  = measurement.clone(),    # y0_batch,
    t_in_ref  = deepcopy(measurement.clone()),    # y0_batch,
    t_out_ref = v_tar_complex,  # x0_batch,
    **adv_param
).detach()

# establish perturbed measurments and rec. images
adversarial_noise = measurement - og_meas
adj_adversarial_noise = OpA.adj(adversarial_noise).norm(p=2,dim=(0,1))
rec = unet.forward(og_meas)
perturbed_rec = unet.forward(measurement)
# from complex to real
img_rec      = rec.norm(p=2, dim=1)[0]
img_pert_rec = perturbed_rec.norm(p=2, dim=1)[0]

# ------------------ Plot adversarial example ------------------
plotdir = os.path.join(config.PLOT_PATH, "adversarial_plots", "supervised", "eta%.1f"%eta)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
save_image(v_tar.detach().cpu(),                                        os.path.join(plotdir, "original_image_small_text.pdf") )
save_image(v_tar.detach().cpu() + adj_adversarial_noise.detach().cpu(), os.path.join(plotdir, "adv_pert_image_small_text.pdf") )
save_image(img_rec.detach().cpu(),                                      os.path.join(plotdir, "rec_small_text.pdf") )
save_image(img_pert_rec.detach().cpu(),                                 os.path.join(plotdir, "adv_pert_rec_small_text.pdf") )
save_image(10*(v_tar - img_rec).abs().detach().cpu(),                   os.path.join(plotdir, "adv_pert_residual_small_text.pdf") )
