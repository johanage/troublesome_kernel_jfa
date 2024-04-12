
# ---------------- Usage ---------------------------------------------------
# sys.argv:
# 1 : index of sampling pattern list of sampling rates
# --------------------------------------------------------------------------
# general imports
import os, sys
import matplotlib as mpl
import torch
import torchvision
from piq import psnr, ssim
# local imports
from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import DeepDecoder
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
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[-1]
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, "circle"), 
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
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

# ----- network configuration -----
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
# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")
def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# set training parameters
num_epochs = 10000 # sampling rate experiment DeepDecoder epoch nr
init_lr = 5e-3
train_params = {
    "num_epochs": num_epochs,
    "batch_size": 1,
    "loss_func": loss_func,
    #"save_path": os.path.join(config.RESULTS_PATH,"DeepDecoder"),
    "save_path": os.path.join(config.SCRATCH_PATH,"DeepDecoder"),
    "save_epochs": num_epochs//10,
    "optimizer_params": {"lr": init_lr, "eps": 1e-8, "weight_decay": 0},
    "scheduler_params": {
        "step_size": num_epochs//100, 
        "gamma"    : 0.98
    },
    "acc_steps": 1,
}

# ------ construct network and train -----
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}_k{dim_channels}_nc{num_channels}_{architecture}".format(
    lr               = init_lr, 
    gamma            = train_params["scheduler_params"]["gamma"],
    sampling_pattern = "%s_sr%.2f"%(sp_type, sampling_rate),
    dim_channels     = dim_channels,
    num_channels     = num_channels,
    architecture     = archkey,
)
# directory of init weights and biases
fn_init_weights = os.path.join(train_params["save_path"],"DeepDecoder_init_weights_%s.pt"%(fn_suffix) )
# if not done before, save the initial deep_decoder weights and biases
if not os.path.isfile(fn_init_weights):
    torch.save(deep_decoder.state_dict(), fn_init_weights)
# load init weights and biases
#init_weights = torch.load(fn_init_weights)
#deep_decoder.load_state_dict(init_weights) 
#del init_weights

if deep_decoder.device == torch.device("cpu"):
    deep_decoder = deep_decoder.to(device)
assert gpu_avail and deep_decoder.device == device, "for some reason DeepDecoder is on %s even though gpu avail %s"%(deep_decoder.device, gpu_avail)
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
sample = to_complex(sample[None])
# simulate measurements by applying the Fourier transform
measurement = OpA(sample)
measurement = measurement[None].to(device)

# Deep decoder input
# height and width of input vector
in_hw = 256//2**(deep_decoder.nscales-1)
input_range = [0,10]
# torch.rand : u ~ U([0,1] so for input_range=[a,b] we get a + b * u ~ U([a,b])
ddinput = input_range[0] + input_range[1] * torch.rand((1, deep_decoder_params["channels_up"][0],in_hw, in_hw))
# save ddinput in case needed for adv. noise study
torch.save(ddinput, os.path.join(train_params["save_path"], "ddinput_%s.pt"%(fn_suffix)) )
ddinput = ddinput.to(device)

# optimizer setup
optimizer = torch.optim.Adam
scheduler = torch.optim.lr_scheduler.StepLR
optimizer = optimizer(deep_decoder.parameters(), **train_params["optimizer_params"])
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
    desc="Train DeepDecoder ",
    total=train_params["num_epochs"],
)

# training loop
isave = 0
###### Save parameters of DeepDecoder model
path = train_params["save_path"]


# prepare reference image
og_complex_img = sample.cpu()[None]
og_img  = og_complex_img.norm(p=2, dim=(0,1))[None, None]

# init figure to plot evolution
from matplotlib import pyplot as plt
num_save = train_params["num_epochs"] // train_params["save_epochs"]
fig, axs = plt.subplots(2,num_save,figsize=(5*num_save,10) )

# magnitude of added gaussian noise during training - noise-based regularisation
sigma_p = 1/50
for epoch in range(train_params["num_epochs"]): 
    deep_decoder.train()  # make sure we are in train mode
    optimizer.zero_grad()
    # noise-based regularisation: add gaussian noise to DeepDecoder input according to Ulyanov et al 2020
    additive_noise = sigma_p*torch.randn(ddinput.shape).to(device)
    model_input    = ddinput + additive_noise
    # pred_img = G(model_input, theta), model_input = ddinput + noise
    pred_img = deep_decoder.forward(model_input)
    # pred = A G(ddinput, theta)
    pred = OpA(pred_img)
    loss = loss_func(pred, measurement)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # compute logging metrics, first prepare predicted image
    deep_decoder.eval()
    with torch.no_grad():
        # get complex reconstruction
        #pred_img_eval = deep_decoder.forward(model_input).cpu()
        pred_img_eval = deep_decoder.forward(ddinput).cpu()
        # compute real image
        img_rec = pred_img_eval.norm(p=2, dim=(0,1))[None, None]
        # Reconstruction error
        rec_err = (og_complex_img - pred_img_eval).norm(p=2)
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
            "rel_rec_err"   : (rec_err.item() / og_complex_img.norm(p=2) ).item(),
            #"loc_lip"       : local_lipshitz_constant.item(),
            "lr"          : scheduler.get_last_lr()[0],
        }, 
        index = [0], 
    )
    logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
    # update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix(
        **deep_decoder._add_to_progress_bar({
        "loss"         : loss.item(), 
        "rel_rec_err"  : app_log["rel_rec_err"][0],
        "rel_eval_diff": app_log["rel_eval_diff"][0],
        })
    )
    if epoch % train_params["save_epochs"] == 0 or epoch == train_params["num_epochs"] - 1:
        print("Saving parameters of models and plotting evolution")
        # set img_rec to 2D shape
        img_rec = img_rec[0,0]
        img     = og_img[0,0]
        
        if epoch < train_params["num_epochs"] - 1:
            # save rec. image
            torch.save(pred_img.detach().cpu(), os.path.join(path, "DeepDecoder_nojit_rec_{suffix}_epoch{epoch}.pt".format(
                suffix = fn_suffix,
                epoch  = epoch,
            )))
            # save the deep_decoder.parameters for each num_epochs
            torch.save(deep_decoder.state_dict(), os.path.join(path,"DeepDecoder_nojit_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) ) )
        else:
            # save last rec img
            torch.save(pred_img.detach().cpu(), os.path.join(path, "DeepDecoder_nojit_rec_{suffix}_epoch{epoch}.pt".format(
                suffix = fn_suffix,
                epoch  = epoch,
            )))
            # save last deep_decoder.params
            torch.save(deep_decoder.state_dict(), os.path.join(path,"DeepDecoder_nojit_{suffix}_last.pt".format(suffix = fn_suffix) ) )

# plot evolution

for epoch in range(train_params["num_epochs"]):
    if epoch % train_params["save_epochs"] == 0:
            # load parameters at epcoch 
            deep_decoder.load_state_dict( torch.load(os.path.join(path,"DeepDecoder_nojit_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) ) ) )
            deep_decoder.eval()
            # reconstruct image at epoch
            #pred_img_eval = deep_decoder.forward(model_input).cpu()
            pred_img_eval = deep_decoder.forward(ddinput).cpu().detach()
            img_rec = pred_img_eval[0].norm(p=2, dim=0)
            # add image eval metrics as text to residual row
            rec_psnr = psnr(img_rec[None, None].clamp(0,1), og_img.detach().cpu())
            rec_ssim = ssim(img_rec[None, None].clamp(0,1), og_img.detach().cpu())
            ###### Plot evolution of training process #######
            cmap = "Greys_r"
            axs[0,isave].imshow(img_rec, cmap=cmap)
            #axs[0,isave].set_title("Epoch %i"%epoch)
            axs[1,isave].imshow((og_img[0,0] - img_rec).abs(), cmap=cmap)
            axs[0,isave].set_axis_off(); axs[1,isave].set_axis_off()
            axs[1,isave].text(x = 5,y = 20, s = "PSNR : %.1f \nSSIM : %.2f"%(rec_psnr, rec_ssim), fontsize = 16)
            isave += 1
        
# save the logging table to pickle
logging.to_pickle(os.path.join(config.RESULTS_PATH, "DeepDecoder", "DeepDecoder_UNet_nojit_logging.pkl"))

# plot evolution of training
fig.tight_layout()
fig_savepath = os.path.join(config.RESULTS_PATH, "../plots/DeepDecoder/")
fig.savefig(os.path.join(fig_savepath, "DeepDecoder_nojit_evolution_{suffix}.png".format(suffix=fn_suffix)), bbox_inches="tight")

# save final reconstruction
deep_decoder.eval()
out_dd_eval = deep_decoder(ddinput)
img_dd_eval = out_dd_eval.norm(p=2, dim=1)
from dip_utils import plot_train_DIP
plot_train_DIP(img, img_rec, logging, save_fn = os.path.join(fig_savepath, "DeepDecoder_nojit_train_metrics_{suffix}.png".format(suffix=fn_suffix)) )
