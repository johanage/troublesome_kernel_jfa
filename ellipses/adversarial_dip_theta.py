from matplotlib import pyplot as plt
import os, sys, torch
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

# ----- Network configuration -----
unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : None, #inverter,
}
unet = UNet

# ----- Training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")
def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# Set adv. training parameters
num_epochs = 1000
init_lr = 5e-4
train_params = {
    "num_epochs": num_epochs,
    "batch_size": 1,
    "loss_func": loss_func,
    "save_path": os.path.join(config.RESULTS_PATH,"DIP"),
    "save_epochs": num_epochs//10,
    "optimizer": torch.optim.Adam,
    "optimizer_params": {"lr": init_lr, "eps": 1e-8, "weight_decay": 0},
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": num_epochs//100, "gamma": 0.96},
    "acc_steps": 1,
}

# ------ construct network and train -----
unet = unet(**unet_params)
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# get train and validation data
dir_train = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/train/"
dir_val = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/val/"

# same as DIP
v_tar = torch.load(dir_val + "sample_00000.pt").to(device)
v_tar_complex = to_complex(v_tar[None]).to(device)
measurement = OpA(v_tar_complex).to(device)[None]

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
noise_mag = .1
# same shape as SR problem in Ulyanov et al 2018
z_tilde = noise_mag * torch.rand((unet_params["in_channels"],) + v_tar.shape)
z_tilde = z_tilde.to(device)[None]

# optimizer setup
dip_optimizer = torch.optim.Adam
dip_optimizer = dip_optimizer(unet.parameters(), **train_params["optimizer_params"])

# load model weights
param_dir = os.getcwd() + "/models/DIP/"
file_param = "DIP_UNet_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1_last.pt"
params_loaded = torch.load(param_dir + file_param)
unet.load_state_dict(params_loaded)
unet.eval()
from find_adversarial import PAdam_DIP
from functools import partial
from operators import proj_l2_ball
# define loss function
# x - target image, y - measurements, net - DL model, adv_noise - adversarial noise
loss_adv = lambda adv_noise, net,z_tilde,y, x, meas_op, beta: ( meas_op(net(z_tilde)) - (y + adv_noise) ).pow(2).sum() - beta*(net(z_tilde) - x).pow(2)
loss_adv_partial = partial(loss_adv, z_tilde = z_tilde, y = measurement, meas_op = OpA)
# init input optimizer (PGD or alternative methods like PAdam)
adv_init_fac = 3
noise_rel = 1e-2
adv_noise_mag = adv_init_fac * noise_rel * measurement.norm(p=2) / np.sqrt(np.prod(measurement.shape[-2:]))
adv_noise_init = adv_noise_mag * torch.randn_like(measurement).to(device)
adv_noise_init.requires_grad = True

# ------------- Projection setup -----------------------------
# radius is the upper bound of the lp-norm of the projeciton operator
radius = torch.tensor(1e-1).to(device)
# centre is here the centre of the measurements - in general the centre of the projection ball 
# centre set to zero freq since measurements are zero-shifted
centre = torch.zeros_like(measurement).to(device)
projection_l2 = partial(proj_l2_ball, radius = radius, centre = centre)


# perform PAdam - uses the ADAM optimizer instead of GD and excludes the backtracking line search
adversarial_noise = PAdam_DIP_theta(
    loss          = loss_adv_partial,
    net           = unet,
    dip_optimizer = dip_optimizer,
    t_in          = adv_noise_init,
    projs         = [projection_l2],
    iter          = 10,
    stepsize      = 1e-4,
    silent        = False,
)

perturbed_measurement = measurement + adversarial_noise
torch.save(perturbed_measurement, os.getcwd() + "/adv_attack_dip/adv_theta_perturbed_measurement.pt")
torch.save(adversarial_noise, os.getcwd() + "/adv_attack_dip/adv_theta_adv_noise.pt")
torch.save(z_tilde, os.getcwd() + "/adv_attack_dip/adv_theta_z_tilde.pt")
