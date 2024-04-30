"""
Training script for supervised learning using a UNet.
UNet is implemented in networks.py.
-----------------------------------------------------
Terms
    - jitter, means that noise, typically Gaussian, is added to the data while training to reduce overfit
"""
# load installed libs
import os
import matplotlib as mpl
import torch
import torchvision
from torchvision.transforms import v2
from itertools import accumulate
import operator
# from local scripts
from data_management import IPDataset, SimulateMeasurements, ToComplex, Jitter
from networks import UNet
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
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
""" Radial sampling golden 180 angle
mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
"""
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[2]
print("sampling rate used is :", sampling_rate)
sp_type = "diamond"
mask_fromfile = MaskFromFile(
    # ---------- a=1 sampling patterns ---------------------------------------------------
    path = os.path.join(config.SP_PATH, sp_type), # circular pattern
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate) # sampling_rate *100 % sr, a = 1, r0 = 2, nlevles = 50 
    # ---------- a=2 sampling patterns ---------------------------------------------------
    #path = config.SP_PATH,
    #filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png" # circular pattern, 25 % sr, a = 2, r0 = 2, nlevels = 50
)
mask = mask_fromfile.mask[None]
# compute sampling rate from mask
sampling_rate_comp = mask.sum().item() / list(accumulate(tuple(mask.shape), operator.mul))[-1]
print("Computed sampling rate: ", sampling_rate_comp)

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
}
unet = UNet
# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

train_phases = 1
num_epochs = [500,0]
lr_init = 1e-4
lr_gamma = 0.98
lr_schedule_stepsize = 60
jitter_params = {"eta" : 10,  "scale_lo" : 0.0, "scale_hi" : 1.0}
train_params = {
    "num_epochs": num_epochs, # fastmri, single-coil
    "batch_size": [10, 10], # ellipses
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            #config.SCRATCH_PATH,
            config.RESULTS_PATH_KADINGIR,
            #"supervised/circ_sr0.25/Fourier_UNet_no_jitter_ellipses_256"
            #"supervised/circ_sr0.25/Fourier_UNet_jitter_mod_brain_fastmri_256"
            #"supervised/circ_sr0.25/Fourier_UNet_jitter_brain_fastmri_256"
            "supervised/%s_sr%.2f/Fourier_UNet_jitter_brain_fastmri_256"%(sp_type, sampling_rate),
            #"supervised/%s_sr%.2f_a2/Fourier_UNet_jitter_brain_fastmri_256"%(sp_type, sampling_rate),
            #"supervised/%s_sr%.2f_a2/Fourier_UNet_jitter_mod_brain_fastmri_256"%(sp_type, sampling_rate),
            #"supervised/%s_sr%.2f/Fourier_UNet_jitter_mod_brain_fastmri_256"%(sp_type, sampling_rate),
            "eta_{eta:0.3f}_train_phase_{train_phase}".format(
                eta = jitter_params["eta"],
                train_phase = (i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 100,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 1e-4, "eps": 2e-4, "weight_decay": 1e-4},
        # lr: start where phase 1 left off
        {"lr": 1e-4*lr_gamma**num_epochs[0]/lr_schedule_stepsize, "eps": 2e-4, "weight_decay": 1e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": lr_schedule_stepsize, "gamma": lr_gamma},
    "acc_steps": [1, 5],
    "train_transform": torchvision.transforms.Compose(
        [
            # flip first -> add imaginary part -> apply meas. operator
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            ToComplex(), # adds an imaginary part with elements set to zero 
            SimulateMeasurements(OpA), # simulate measurments with operator OpA - for MRI its the DFT
            # jitter: add noise
            Jitter(**jitter_params),
        ]
    ),
    "val_transform": torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)],
    ),
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}
# ----- data configuration -----
train_data_params = {
    "path": config.DATA_PATH,
}
train_data = IPDataset

val_data_params = {
    "path": config.DATA_PATH,
}
val_data = IPDataset

# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in unet_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
unet = unet(**unet_params)

# Regular jittering : start from previously trained network
# Modified jittering : start from pre-trained high noise jittering network
"""
param_dir = "supervised/circle_sr0.25_a2/Fourier_UNet_jitter_brain_fastmri_256/eta_100.000_train_phase_1"
#param_dir = "supervised/circle_sr0.25_a2/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_50.000_train_phase_1"
file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(config.RESULTS_PATH_KADINGIR, param_dir,file_param))
unet.load_state_dict(params_loaded)
#"""

if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)
# get train and validation data
# data has shape (number of samples, (measurements, images) )
# Note that the second dimension consist of a 2-tuple
# image x has shape (2, N, N), since x in C^{N x N}
# measurement y has shape (2, m) since y in C^m
train_data = train_data("train", **train_data_params)
val_data = val_data("val", **val_data_params)
# run training
mod = False # activate/deactivate modified jittering: low noise -> no noise
for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )
    if i == 1 and mod: 
        jit_trans = train_params_cur["train_transform"].transforms.pop() # remove jittering
        assert isinstance(jit_trans, Jitter), "popped the wrong transform, removed %s"%(type(jit_trans))
    # print train params
    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))
    unet.train_on(train_data, val_data, **train_params_cur)
