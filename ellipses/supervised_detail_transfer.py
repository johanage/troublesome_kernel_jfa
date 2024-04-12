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
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
# from local scripts
from data_management import (
    IPDataset, 
    ToComplex, 
    AddDetail,
    SimulateMeasurements, 
    Jitter,
)
from networks import UNet
from operators import (
    to_complex,
    Fourier,
    Fourier_matrix as Fourier_m,
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
sampling_rate = 0.25
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, sp_type), # circular pattern
    #path = config.SP_PATH,
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate) # sampling_rate *100 % sr, a = 1, r0 = 2, nlevles = 50 
    #filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png" # circular pattern, 25 % sr, a = 2, r0 = 2, nlevels = 50
)
mask = mask_fromfile.mask[None]

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
    #"upsampling"    : "nearest",
    "upsampling"    : "trans_conv",
}
# ------ construct network and train -----
unet = UNet(**unet_params)
# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# Details to be added
# set directories
savedir     = os.path.join(config.DATA_PATH, "train")
plotdir     = os.path.join(config.PLOT_PATH, "detail_transfer")

# ------ Load small detail to check if in/close to nullspace of A --------
nullspace_data_det = torch.load(os.path.join(plotdir, "detail.pt") )

# ------ Load the image with only the large detail: x + x_large_det --------------
large_det = torch.load(os.path.join(plotdir, "large_detail.pt"))

train_phases = 0
#num_epochs = [500,100]
num_epochs = [300,0]
lr_gamma = 0.98
lr_schedule_stepsize = 10
jitter_params = {"eta" : 1e-1,  "scale_lo" : 0.0, "scale_hi" : 1.0}
train_params = {
    "num_epochs": num_epochs, # fastmri, single-coil
    "batch_size": [10, 10], # ellipses
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH_KADINGIR,
            "detail_transfer",
            "supervised/%s_sr%.2f/upsampling_%s/Fourier_UNet_jitter_mod_brain_fastmri_256"%(sp_type, sampling_rate, unet_params["upsampling"]),
            #"eta_{eta:0.3f}_train_phase_{train_phase}".format(
            "eta_{eta:0.3f}_train_phase_{train_phase}_dataconsistency".format(
                eta = jitter_params["eta"],
                train_phase = (i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 100,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 1e-4, "eps": 2e-4, "weight_decay": 1e-4},
        # start where phase 1 left off
        {"lr": 1e-4**(num_epochs[0]/lr_schedule_stepsize), "eps": 2e-4, "weight_decay": 1e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": lr_schedule_stepsize, "gamma": lr_gamma},
    "acc_steps": [1, 1],#200],
    "train_transform": torchvision.transforms.Compose(
        [
            # include imaginary part -> add details -> invariant transforms -> apply meas. operator
            ToComplex(), # adds an imaginary part with elements set to zero 
            AddDetail(idx=42, detail = nullspace_data_det + to_complex(large_det[None,None])), # add nullspace detail and large detail 
            #v2.RandomHorizontalFlip(p=0.5),
            #v2.RandomVerticalFlip(p=0.5),
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

# Regular jittering : start from previously trained network
# Modified jittering : start from pre-trained high noise jittering network
#"""
param_dir = "detail_transfer/supervised/circle_sr0.25/upsampling_trans_conv/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_0.100_train_phase_2/"
file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(config.RESULTS_PATH_KADINGIR,param_dir,file_param))
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
# just include the sample where the details have been added
train_data.files = [x for x in train_data.files if "00042" in x]
val_data = val_data("val", **val_data_params)
# run training
mod = True # activate/deactivate modified jittering: low noise -> no noise
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

######## Reconstruction #######################################
#param_dir = "detail_transfer/supervised/circle_sr0.25/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_0.100_train_phase_1/"
#param_dir = "detail_transfer/supervised/circle_sr0.25/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_0.100_train_phase_2/"
#param_dir = "detail_transfer/supervised/circle_sr0.25/upsampling_trans_conv/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_0.100_train_phase_2/"
param_dir = "detail_transfer/supervised/circle_sr0.25/upsampling_trans_conv/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_0.100_train_phase_1_dataconsistency/"
#file_param = "model_weights_epoch400.pt"
file_param = "model_weights.pt"
params_loaded = torch.load(os.path.join(config.RESULTS_PATH_KADINGIR, param_dir,file_param))
unet.load_state_dict(params_loaded)
unet.eval()
for p in unet.parameters():
    p.requires_grad = False

# ------- Load the image: x  --------------------------------------------------
idx = "00042"
loadfn = os.path.join(savedir, "sample_{idx}.pt".format(idx=idx))
data = torch.load(loadfn)

# ------ Load small detail to check if in/close to nullspace of A --------
nullspace_data_det = torch.load(os.path.join(plotdir, "detail.pt") )
measurement_nullspace_det = OpA(to_complex(nullspace_data_det))

# ------ Load the image with only the large detail: x + x_large_det --------------
data_large_det = torch.load(os.path.join(plotdir, "image_plus_large_detail.pt"))
large_det = torch.load(os.path.join(plotdir, "large_detail.pt"))

# ------- Compute the detailed image -----------------
data_detailed_image = to_complex(data[None,None]) + nullspace_data_det + to_complex(large_det[None,None])
im_det = data_detailed_image.norm(p=2,dim=(0,1))
# ---- Double check correct composition om image -------------------------------------------------------
datalist = [data, im_det, data_large_det, 
    (data_detailed_image - to_complex(data[None,None])).norm(p=2,dim=(0,1)), 
    data_large_det - data, 
    (data_detailed_image - to_complex(data_large_det[None,None])).norm(p=2,dim=(0,1))]
fig, axs = plt.subplots(1,len(datalist), figsize=(20,4))
cmap = "Greys_r"
plots_dc = [axs[i].imshow(datalist[i], cmap=cmap) for i in range(len(datalist))]
for ax,plot in zip(axs,plots_dc):
    divider = mal(ax)
    cax     = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(plot, cax=cax)
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "double_check_images.png"), dpi=140)

# --- Compute measurements ----------------------------------------------
measurement_detailed_image = OpA(data_detailed_image)
measurement_large_det      = OpA(to_complex(data_large_det[None,None]))
measurement_sample         = OpA(to_complex(data[None,None]))
# --- Reconstruct images  -----------------------------------------------
rec_det_im    = unet.forward(measurement_detailed_image.to(device))
rec_large_det = unet.forward(measurement_large_det.to(device))
rec_im        = unet.forward(measurement_sample.to(device))
# --- Plot reconstructions ----------------------------------------------
reclist = [data_detailed_image, rec_det_im, rec_large_det, rec_im]
fig, axs = plt.subplots(1,len(reclist), figsize=(20,4))
cmap = "Greys_r"
plots_rec = [axs[i].imshow(reclist[i].detach().cpu().norm(p=2, dim=(0,1)), cmap=cmap) for i in range(len(reclist))]
for ax,plot in zip(axs,plots_rec):
    divider = mal(ax)
    cax     = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(plot, cax=cax)
fig.tight_layout()
