import os, torch, torchvision
import matplotlib as mpl
from piq import psnr, ssim

from data_management import IPDataset, SimulateMeasurements, ToComplex, Jitter
from networks import UNet
from operators import (
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
#mask_func = RadialMaskFunc(config.n, 40)
#mask = mask_func((1,) + config.n + (1,))
#mask = mask.squeeze(-1)
#mask = mask.unsqueeze(1)
mask_fromfile = MaskFromFile(
    path = os.getcwd() + "/sampling_patterns/", 
    filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png"
)
# Fourier matrix
mask = mask_fromfile.mask[None]
OpA_m = Fourier_m(mask)
# Fourier operator
OpA = Fourier(mask)
inverter = LearnableInverterFourier(config.n, mask, learnable=False)
# set device for operators
OpA_m.to(device)
inverter.to(device)

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")
def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# set training parameters
num_epochs = 10000
init_lr = 5e-4
train_params = {
    "num_epochs": num_epochs,
    "batch_size": 1,
    "loss_func": loss_func,
    "save_path": os.path.join(config.RESULTS_PATH,"DIP"),
    "save_epochs": num_epochs//100,
    #"optimizer": torch.optim.Adam,
    "optimizer_params": {"lr": init_lr, "eps": 1e-8, "weight_decay": 0},
    #"scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": num_epochs//100, "gamma": 0.96},
    "acc_steps": 1,
}

# ----- network configuration -----
unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : None,
    "upsampling"    : "nearest",
}
# ------ construct network and train -----
unet = UNet(**unet_params)

# save the initial unet weights and biases
fn_init_weights = os.path.join(train_params["save_path"],"DIP_UNet_init_weights.pt" )
init_weights = torch.load(fn_init_weights)
unet.load_state_dict(init_weights)
breakpoint()

# load unet to GPU
if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# get train and validation data
# Vegard's scratch folder
#dir_train = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/train/"
#dir_val = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/val/"
# JFA's local dir
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val   = os.path.join(config.DATA_PATH, "val")
# NOTE: both Vegard's and JFA's dirs does not contain test dir
# Load one sample to train network on
sample = torch.load(os.path.join(dir_train,"sample_00000.pt"))
# sample is real valued so make fake imaginary part
sample = sample[None].repeat(2,1,1)
# set imaginary values to zero
sample[1]   = torch.zeros_like(sample[1])
# simulate measurements by applying the Fourier transform
measurement = OpA(sample)
measurement = measurement[None].to(device)

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
noise_mag = .1
# same shape as SR problem in Ulyanov et al 2018
z_tilde = noise_mag * torch.rand((1, unet_params["in_channels"],) + tuple(sample.shape[1:]))
z_tilde = z_tilde.to(device)

# optimizer setup
optimizer = torch.optim.Adam
scheduler = torch.optim.lr_scheduler.StepLR
optimizer = optimizer(unet.parameters(), **train_params["optimizer_params"])
scheduler = scheduler(optimizer, **train_params["scheduler_params"])

# log setup
import pandas as pd
logging = pd.DataFrame(
    columns=["loss", "lr", "psnr", "ssim"]
)
# progressbar setup
from tqdm import tqdm
progress_bar = tqdm(
    desc="Train DIP ",
    total=train_params["num_epochs"],
)
from matplotlib import pyplot as plt
num_save = train_params["num_epochs"] // train_params["save_epochs"]
fig, axs = plt.subplots(2,num_save,figsize=(5*num_save,10) )

# function that returns img of sample and the reconstructed image
from dip_utils import get_img_rec, center_scale_01

# jittering setup
jit_params = {"eta" : 1/10,  "scale_lo" : 0.0, "scale_hi" : 1.0}
Jitter_trans = Jitter(**jit_params)
noisy_measurement, target = Jitter_trans( (measurement, sample[None]) )

###### Save parameters of DIP model
path = train_params["save_path"]
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}".format(
    lr = init_lr, 
    gamma = train_params["scheduler_params"]["gamma"],
    sampling_pattern = "circ_sr2.5e-1",
)
torch.save((noisy_measurement - measurement).detach().cpu(), os.path.join(path, "DIP_jit_eta_{eta}_perturbation_{suffix}.pt".format(
    eta    = jit_params["eta"],
    suffix = fn_suffix,
)))

# training loop
isave = 0
# magnitude of added gaussian noise during training
sigma_p = 1/30
for epoch in range(train_params["num_epochs"]): 
    unet.train()  # make sure we are in train mode
    optimizer.zero_grad()
    # add gaussian noise to DIP input according to Ulyanov et al 2020
    additive_noise = sigma_p*torch.randn(z_tilde.shape).to(device)
    model_input = z_tilde + additive_noise
    # get img = Re(sample), img_rec = Re(pred_img), pred_img = G(z_tilde, theta)
    img, img_rec, pred_img = get_img_rec(sample, model_input, model = unet)
    # pred = A G(z_tilde, theta)
    pred = OpA(pred_img)
    #loss = loss_func(pred, measurement)
    loss = loss_func(pred, noisy_measurement)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # compute logging metrics, first prepare predicted image
    unet.eval()
    img, img_rec, pred_img = get_img_rec(sample, z_tilde, model = unet)
    ssim_pred = ssim( img[None,None], center_scale_01(image = img_rec)[None,None] )
    psnr_pred = psnr( img[None,None], center_scale_01(image = img_rec)[None,None] )
    # append to log
    app_log = pd.DataFrame( 
        {
            "loss" : loss.item(), 
            "lr"   : scheduler.get_last_lr()[0],
            "psnr" : psnr_pred,
            "ssim" : ssim_pred,
        }, 
        index = [0], 
    )
    logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
    
    # update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix(
        **unet._add_to_progress_bar({"loss": loss.item()})
    )
    if epoch % train_params["save_epochs"] == 0 or epoch == train_params["num_epochs"] - 1:
        print("Saving parameters of models and plotting evolution")
        
        if epoch < train_params["num_epochs"] - 1:
            # save rec. image
            torch.save(pred_img.detach().cpu(), os.path.join(path, "DIP_jit_eta_{eta}_rec_{suffix}_epoch{epoch}.pt".format(
                eta    = jit_params["eta"],
                suffix = fn_suffix,
                epoch  = epoch,
            )))
            # save perturbation
            torch.save((noisy_measurement - measurement).detach().cpu(), os.path.join(path, "DIP_jit_eta_{eta}_perturbation_{suffix}_epoch{epoch}.pt".format(
                eta    = jit_params["eta"],
                suffix = fn_suffix,
                epoch  = epoch,
            )))
            # save the unet parameters for each num_epochs
            torch.save(unet.state_dict(), os.path.join(path,"DIP_UNet_jit_eta_{eta}_{suffix}_epoch{epoch}.pt".format(
                eta    = jit_params["eta"],
                suffix = fn_suffix, 
                epoch  = epoch, 
            ) ) )
            ###### Plot evolution of training process #######
            cmap = "Greys_r"
            axs[0,isave].imshow(img_rec, cmap=cmap)
            #axs[0,isave].set_title("Epoch %i"%epoch)
            axs[1,isave].imshow(.5*torch.log( (img - img_rec)**2), cmap=cmap)
            axs[0,isave].set_axis_off(); axs[1,isave].set_axis_off()
            isave += 1       
        else:
            # save rec. image
            torch.save(pred_img.detach().cpu(), os.path.join(path, "DIP_jit_eta_{eta}_rec_{suffix}_epoch{epoch}.pt".format(
                eta    = jit_params["eta"],
                suffix = fn_suffix,
                epoch  = epoch,
            )))
            # save perturbation
            torch.save((noisy_measurement - measurement).detach().cpu(), os.path.join(path, "DIP_jit_eta_{eta}_perturbation_{suffix}_epoch{epoch}.pt".format(
                eta    = jit_params["eta"],
                suffix = fn_suffix,
                epoch  = epoch,
            )))
            # save last unet params
            torch.save(unet.state_dict(), os.path.join(path,"DIP_UNet_jit_eta_{eta}_{suffix}_last.pt".format(eta = jit_params["eta"], suffix = fn_suffix) ) )
     

# TODO make figures presentable and functions where it is necessary
fig.tight_layout()
fig_savepath = os.path.join(config.RESULTS_PATH, "..", "plots/DIP/")
fig.savefig(os.path.join(fig_savepath, "DIP_jit_evolution.png"), bbox_inches="tight")

# save final reconstruction
unet.eval()
img, img_rec, rec = get_img_rec(sample, z_tilde, model = unet) 
# center and normalize to x_hat in [0,1]
img_rec = (img_rec - img_rec.min() )/ (img_rec.max() - img_rec.min() )
from dip_utils import plot_train_DIP
plot_train_DIP(img, img_rec, logging, save_fn = os.path.join(fig_savepath,"DIP_jit_train_metrics.png") )
