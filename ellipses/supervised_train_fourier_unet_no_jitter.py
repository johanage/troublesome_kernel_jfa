# load installed libs
import os
import matplotlib as mpl
import torch
import torchvision
from torchvision.transforms import v2
from itertools import accumulate
import operator

# from local scripts
from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import UNet
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    to_complex,
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
#"""
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[2]
print("Sampling rate used is (only correct for multilevel sampling patterns): ", sampling_rate)
sp_type = "circle"
# NOTE: emprically observed that the a=2 circular pattern gives ~1% less rel. l2-error
mask_fromfile = MaskFromFile(
    # ------------ a=1 sampling patterns ----------------
    path = os.path.join(config.SP_PATH, sp_type), # circular pattern
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate) # sampling_rate *100 % sr, a = 1, r0 = 2, nlevles = 50 
    # ------------ a=2 sampling patterns ----------------
    #path = config.SP_PATH,
    #filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png" # circular pattern, 25 % sr, a = 2, r0 = 2, nlevels = 50
)
mask = mask_fromfile.mask[None]
# compute sampling rate from mask
sampling_rate_comp = mask.sum().item() / list(accumulate(tuple(mask.shape), operator.mul))[-1]
print("Computed sampling rate is: ", sampling_rate_comp)
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
num_epochs = [200, 0] # fastMRI
#num_epochs = [10, 0]   # ellipses
lr_gamma = 0.98
lr_scheduler_step = 10
a_sampling_pattern = 1 # a parameter of the multilevel sampling pattern for more informed dir names 
train_params = {
    "num_epochs": num_epochs, # fastmri, single-coil
    "batch_size" : [10, 10], 
    "loss_func"  : loss_func,
    "save_path"  : [
        os.path.join(
            #config.SCRATCH_PATH,  
            config.RESULTS_PATH_KADINGIR,  
            #"supervised/ellipses/%s_sr%.2f/Fourier_UNet_no_jitter_ellipses_256"%(sp_type, sampling_rate),
            "supervised/%s_sr%.2f_a%i/Fourier_UNet_no_jitter_brain_fastmri_256"%(sp_type, sampling_rate, a_sampling_pattern),
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs"      : 100,
    "optimizer"        : torch.optim.Adam,
    "optimizer_params" : [
        {"lr": 1e-4, "eps": 2e-4, "weight_decay": 1e-4},
        # start where phase 1 left off
        {"lr": 1e-4*lr_gamma**num_epochs[0], "eps": 2e-4, "weight_decay": 1e-4},
    ],
    "scheduler"        : torch.optim.lr_scheduler.StepLR,
    "scheduler_params" : {"step_size": lr_scheduler_step, "gamma": lr_gamma},
    "acc_steps"        : [1, 1],
    "train_transform"  : torchvision.transforms.Compose(
        [
            # flip first -> add imaginary part -> apply meas. operator
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            ToComplex(), # adds an imaginary part with elements set to zero 
            SimulateMeasurements(OpA), # simulate measurments with operator OpA - for MRI its the DFT
        ]
    ),
    "val_transform"       : torchvision.transforms.Compose(
        [ToComplex(), SimulateMeasurements(OpA)],
    ),
    "train_loader_params" : {"shuffle": True, "num_workers": 8},
    "val_loader_params"   : {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----
datapath = config.DATA_PATH # fastMRI
#datapath = config.TOY_DATA_PATH # ellipses
train_data_params = {
    "path": datapath,
}
train_data = IPDataset

val_data_params = {
    "path": datapath,
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
# set device for unet
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)


# start from previously trained network
#"""
#param_dir = "supervised/circle_sr0.25_a2/Fourier_UNet_jitter_brain_fastmri_256/eta_0.100_train_phase_1" # low noise jitter SL UNet
#param_dir = "supervised/%s_sr%.2f/Fourier_UNet_no_jitter_brain_fastmri_256/train_phase_%i"%(sp_type, sampling_rate, 1)
#param_dir = "supervised/%s_sr%.2f/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_%.3f_train_phase_%i"%(sp_type, sampling_rate, 0.1, 1)
#param_dir = "supervised/%s_sr%.2f/Fourier_UNet_no_jitter_brain_fastmri_256_single_sample/train_phase_1"%(sp_type, sampling_rate)
param_dir = "supervised/%s_sr%.2f/Fourier_UNet_jitter_brain_fastmri_256/eta_%.3f_train_phase_%i"%(sp_type, sampling_rate, 0.1, 1)
file_param    = "model_weights.pt"
#params_loaded = torch.load(os.path.join(config.SCRATCH_PATH, param_dir, file_param) )
params_loaded = torch.load(os.path.join(config.RESULTS_PATH_KADINGIR, param_dir, file_param) )
unet.load_state_dict(params_loaded)
#"""

# reconstruct val sample with added text "CANCER"
"""unet.eval()
with torch.no_grad():
    sample = to_complex(torch.load(os.path.join(datapath, "train", "sample_00000.pt"))[None, None].to(device))
    rec = unet.forward(OpA(sample)).cpu()
    #sample_text = to_complex(torch.load(os.path.join(datapath, "val", "sample_00042_text.pt"))[None, None].to(device))
    #rec_cancer = unet.forward(OpA(sample_text)).cpu()
#torchvision.utils.save_image(rec_cancer.norm(p=2,dim=(0,1)), os.path.join(config.PLOT_PATH, "supervised", "fig_example_S00042_adv_unet_no_jit_text.pdf" ) )
torchvision.utils.save_image(rec.norm(p=2,dim=(0,1)), os.path.join(config.PLOT_PATH, "supervised", "fig_example_S00042_adv_unet_no_jit_text.pdf" ) )
"""
# get train and validation data
# data has shape (number of samples, (measurements, images) )
# Note that the second dimension consist of a 2-tuple
# image x has shape (2, N, N), since x in C^{N x N}
# measurement y has shape (2, m) since y in C^m
train_data = train_data("train", **train_data_params)
val_data = val_data("val", **val_data_params)

# --- Set train_data to single sample ----------
#sample_idx = 0
#train_data.files = [x for x in train_data.files if "%.5i"%sample_idx in x]

# run training 
for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )
    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))
    unet.train_on(train_data, val_data, **train_params_cur)
