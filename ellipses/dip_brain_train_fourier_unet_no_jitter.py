
# general imports
import os, sys
import numpy as np
import matplotlib as mpl
import torch
from torchvision.utils import save_image
from piq import psnr, ssim
from itertools import accumulate
import operator
# --------------------------------------------------------------------------
# local imports
from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import UNet, local_lipshitz
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
"""
mask_func = RadialMaskFunc(config.n, 17)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
#"""
# ----- import from file ----------------------------------
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[2]
#sampling_rate = sr_list[int(sys.argv[1])]
#sampling_rate = mask.sum().item() / list(accumulate(tuple(mask.shape), operator.mul))[-1]
sp_type = "diamond"
print("sampling pattern : ", sp_type, "sampling rate used is :", sampling_rate)
mask_fromfile = MaskFromFile(
    # -------- a=1 Samplig patterns --------------------
    path = os.path.join(config.SP_PATH, "%s"%sp_type), 
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
    # -------- a=2 Samplig patterns --------------------
    #path = config.SP_PATH, 
    #filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png",
    #filename = "multilevel_sampling_pattern_diamond_sr2.500000e-01_a2_r0_2_levels50.png",
)
mask = mask_fromfile.mask[None]
# compute the sampling rate from the mask 
sampling_rate_comp = mask.sum().item() / list(accumulate(tuple(mask.shape), operator.mul))[-1]
print("Computed sampling rate is: ", sampling_rate_comp)

# Fourier matrix
OpA_m = Fourier_m(mask)
# Fourier operator
OpA = Fourier(mask)
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
    "inverter"      : None,
    "upsampling"    : "nearest",
}
unet = UNet
# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")
def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# set training parameters
num_epochs = 20000 # sampling rate experiment DIP epoch nr
init_lr = 1e-4
train_params = {
    "num_epochs": num_epochs,
    "batch_size": 1,
    "loss_func": loss_func,
    #"save_path": os.path.join(config.RESULTS_PATH,"DIP"),
    #"save_path": os.path.join(config.SCRATCH_PATH,"DIP"),
    #"save_path": os.path.join(config.SCRATCH_PATH,"DIP", "ellipses"),
    #"save_path": os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "sampling_rate_experiment"),
    #"save_path": os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "adv_attacks", "orig_rec", "noiseless_meas", "%iiter"%num_epochs),
    #"save_path": os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "a2"),
    "save_path": os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "a1"),
    "save_epochs": num_epochs//10,
    #"optimizer": torch.optim.Adam,
    "optimizer_params": {"lr": init_lr, "eps": 1e-8, "weight_decay": 0},
    #"scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size": 100, 
        "gamma"    : 0.99,
    },
    "acc_steps": 1,
}

# ------ construct network and train -----
unet = unet(**unet_params)

# ---------- init weights and biases ------------
fn_init_weights = os.path.join(train_params["save_path"],"DIP_UNet_init_weights_%s_%.2f.pt"%(sp_type, sampling_rate) )
# if not done before, save the initial unet weights and biases
# ---------- save weights and biases -------------
if not os.path.isfile(fn_init_weights):
    torch.save(unet.state_dict(), fn_init_weights)

# ---------- load init weights and biases -----------
init_weights = torch.load(fn_init_weights)
unet.load_state_dict(init_weights) 
del init_weights

if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# get train and validation data
# set data directories
dir_train = os.path.join(config.DATA_PATH, "train")
#dir_train = os.path.join(config.TOY_DATA_PATH, "train")
dir_val   = os.path.join(config.DATA_PATH, "val")
#dir_val   = os.path.join(config.TOY_DATA_PATH, "val")

# Load one sample to train network on
sample_idx = 0 
sample = torch.load(os.path.join(dir_train,"sample_%.5i.pt"%sample_idx)) # sample from train set
#sample = torch.load(os.path.join(dir_val, "sample_%.5i_text.pt"%sample_idx) ) # sample 21 or 42 with text details
#sample = torch.load(os.path.join(dir_train,"sample_%i.pt"%sample_idx)) # sample from ellipses dataset
sample = to_complex(sample[None,None])
# simulate measurements by applying the Fourier transform
measurement = OpA(sample)
measurement = measurement.to(device)
#meas_noise_std = 0.08 * measurement.norm(p=2) / np.sqrt(np.prod( measurement.shape[-2:] ) )
#meas_noise = meas_noise_std * torch.randn_like(measurement)
#measurement += meas_noise 
#print(" l2-norm of Gaussaian noise : ", meas_noise.norm(p=2)/ OpA(sample).norm(p=2) ) 


save_ztilde = True
if save_ztilde:
    # init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
    noise_mag = .1
    # same shape as SR problem in Ulyanov et al 2018
    z_tilde = noise_mag * torch.rand((1, unet_params["in_channels"],) + tuple(sample.shape[-2:]))
    # save z_tilde in case needed for adv. noise study
    torch.save(z_tilde, os.path.join(train_params["save_path"], "z_tilde_%s_%.2f.pt"%(sp_type, sampling_rate)) )
# load z_tilde
else:
    z_tilde = torch.load(os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "sampling_rate_experiment", "z_tilde_%s_%.2f.pt"%(sp_type, sampling_rate)) )
z_tilde = z_tilde.to(device)

# optimizer setup
optimizer = torch.optim.Adam
scheduler = torch.optim.lr_scheduler.StepLR
optimizer = optimizer(unet.parameters(), **train_params["optimizer_params"])
scheduler = scheduler(optimizer, **train_params["scheduler_params"])

# log setup
import pandas as pd
logging = pd.DataFrame(
    columns=[
        "loss", 
        "lr", 
        "psnr", 
        "ssim", 
        "loc_lip"
    ]
)
# progressbar setup
from tqdm import tqdm
progress_bar = tqdm(
    desc="Train DIP ",
    total=train_params["num_epochs"],
)

# function that returns img of sample and the reconstructed image
from dip_utils import get_img_rec#, center_scale_01

# training loop
isave = 0
###### Save parameters of DIP model
path = train_params["save_path"]
fn_suffix = "lr_{lr}_gamma_{gamma}_step_{gamma_step}_sp_{sampling_pattern}{additional}".format(
    lr               = init_lr, 
    gamma            = train_params["scheduler_params"]["gamma"],
    gamma_step       = train_params["scheduler_params"]["step_size"],
    sampling_pattern = "%s_sr%.2f"%(sp_type, sampling_rate),
    additional       = "_a1",
    #additional       = "_a2",
    #additional       = "_a2_adv%iiter"%num_epochs,
)

# prepare reference image
og_img  = sample.norm(p=2, dim=(0,1))[None, None]

# init figure to plot evolution
from matplotlib import pyplot as plt
num_save = train_params["num_epochs"] // train_params["save_epochs"]
fig, axs = plt.subplots(2,num_save,figsize=(5*num_save,10) )

# magnitude of added gaussian noise during training - noise-based regularisation
sigma_p = 1/30
reconstruct = True#False
if reconstruct:
    for epoch in range(train_params["num_epochs"]): 
        unet.train()  # make sure we are in train mode
        optimizer.zero_grad()
        # noise-based regularisation: add gaussian noise to DIP input according to Ulyanov et al 2020
        additive_noise = sigma_p*torch.randn(z_tilde.shape).to(device)
        model_input    = z_tilde + additive_noise
        # pred_img = G(model_input, theta), model_input = z_tilde + noise
        pred_img = unet.forward(model_input)
        # pred = A G(z_tilde, theta)
        pred = OpA(pred_img)
        loss = loss_func(pred, measurement)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # compute logging metrics, first prepare predicted image
        with torch.no_grad():
            unet.eval()
            # get complex reconstruction
            pred_img_eval = unet.forward(z_tilde).cpu()
            # compute real image
            img_rec = pred_img_eval.norm(p=2, dim=(0,1))[None,None]
            # Reconstruction error
            rec_err = (sample - pred_img_eval).norm(p=2)
            # SSIM
            ssim_pred = ssim( og_img, img_rec.clamp(0,1))
            # PSNR
            psnr_pred = psnr( og_img, img_rec.clamp(0,1))
            # compute the difference between the complex image in train mode vs. eval mode
            rel_eval_diff = (torch.log( (pred_img.detach().cpu() - pred_img_eval).norm(p=2)/pred_img.norm(p=2) ) / torch.log(torch.tensor(10)) )
        # append to log
        app_log = pd.DataFrame( 
            {
                "loss"          : loss.item(), 
                "rel_eval_diff" : rel_eval_diff.item(),
                "psnr"          : psnr_pred.item(),
                "ssim"          : ssim_pred.item(),
                "rec_err"       : rec_err.item(),
                "rel_rec_err"   : (rec_err.item() / sample.norm(p=2) ).item(),
                "lr"          : scheduler.get_last_lr()[0],
            }, 
            index = [0], 
        )
        logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
        
        # update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(
            **unet._add_to_progress_bar({
            "loss"         : loss.item(), 
            "rel_rec_err"  : app_log["rel_rec_err"][0],
            "rel_eval_diff": app_log["rel_eval_diff"][0],
            })
        )
        #if epoch > 10000 and (img_rec - img).norm(p=2)/img.norm(p=2) > 10**-.8:
        #    print("spike in rel. l2-err!")
        #    torch.save(pred_img.detach().cpu(), os.path.join(path, "DIP_nojit_rec_{suffix}_epoch{epoch}.pt".format(
        #        suffix = fn_suffix,
        #        epoch  = epoch,
        #    )))
        # save reconstruction and network weigths every save_epoch
        if epoch % train_params["save_epochs"] == 0 or epoch == train_params["num_epochs"] - 1:
            print("Saving parameters of models and plotting evolution")
            # set img_rec to 2D shape
            img_rec = img_rec[0,0]
            img     = og_img[0,0]
            
            if epoch < train_params["num_epochs"] - 1:
                # save rec. image
                torch.save(pred_img.detach().cpu(), os.path.join(path, "DIP_nojit_rec_{suffix}_epoch{epoch}.pt".format(
                    suffix = fn_suffix,
                    epoch  = epoch,
                )))
                # save the unet parameters for each num_epochs
                torch.save(unet.state_dict(), os.path.join(path,"DIP_UNet_nojit_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) ) )
                
            else:
                # save last rec img
                torch.save(pred_img.detach().cpu(), os.path.join(path, "DIP_nojit_rec_{suffix}_last.pt".format(
                    suffix = fn_suffix,
                )))
                # save last unet params
                torch.save(unet.state_dict(), os.path.join(path,"DIP_UNet_nojit_{suffix}_last.pt".format(suffix = fn_suffix) ) )

# save the logging table to pickle
save_logging = True#False
if save_logging:
    logging.to_pickle(os.path.join(path, "DIP_UNet_nojit_logging_{suffix}.pkl".format(suffix = fn_suffix)))
# load logging
else:
    logging = pd.read_pickle(os.path.join(path, "DIP_UNet_nojit_logging_{suffix}.pkl".format(suffix = fn_suffix)))

# plot evolution
for epoch in range(train_params["num_epochs"]):
    if epoch % train_params["save_epochs"] == 0:
            # load reconstruction
            img_rec = torch.load(os.path.join(path,"DIP_nojit_rec_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) ) )
            img_rec = img_rec.norm(p=2, dim=(0,1)) 
            ###### Add image eval metrics as text to residual row ###
            rec_psnr = logging["psnr"][epoch] #psnr(img_rec[None, None].clamp(0,1), og_img.detach().cpu())
            rec_ssim = logging["ssim"][epoch] #ssim(img_rec[None, None].clamp(0,1), og_img.detach().cpu())
            ###### Plot evolution of training process #######
            cmap = "Greys_r"
            axs[0,isave].imshow(img_rec, cmap=cmap, vmin=0, vmax=1)
            save_image(img_rec, os.path.join(config.PLOT_PATH, "DIP", "evolution", "DIP_nojit_epoch{epoch}_{suffix}.png".format(
                suffix = fn_suffix,
                epoch  = epoch,
            )) )

            #axs[0,isave].set_title("Epoch %i"%epoch)
            axs[1,isave].imshow((og_img[0,0] - img_rec).abs(), cmap=cmap, vmin=0, vmax=.2)
            axs[1,isave].text(x = 5,y = 60, s = "PSNR : %.1f \nSSIM : %.2f"%(rec_psnr, rec_ssim), fontsize = 36, color="white")
            axs[0,isave].set_axis_off(); axs[1,isave].set_axis_off()
            isave += 1
# save plot of the evolution        
fig.tight_layout()
fig_savepath = os.path.join(config.PLOT_PATH, "DIP", "sample_rate_experiment")
fig.savefig(os.path.join(fig_savepath, "DIP_nojit_evolution_{suffix}.png".format(suffix=fn_suffix)), bbox_inches="tight")

# load the final reconstruction
img_rec = torch.load(os.path.join(path, "DIP_nojit_rec_{suffix}_last.pt".format(suffix = fn_suffix) ) )
# save the final reconstruction
save_image(img_rec.norm(p=2,dim=(0,1)), os.path.join(fig_savepath, "DIP_nojit_rec_{suffix}_last.png".format(suffix=fn_suffix) ) )
from dip_utils import plot_train_DIP
plot_train_DIP(sample.norm(p=2, dim=(0,1)), img_rec.norm(p=2, dim=(0,1)), logging, save_fn = os.path.join(fig_savepath, "DIP_nojit_train_metrics_{suffix}.png".format(suffix=fn_suffix)) )
