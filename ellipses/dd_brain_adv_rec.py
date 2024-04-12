# python libs imports
import os
import matplotlib as mpl
import torch
import torchvision
from piq import psnr, ssim
# local imports
from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import DeepDecoder
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
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[-1]
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, "circle"),
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
)

# Fourier matrix
OpA_m = Fourier_m(mask_fromfile.mask[None])
# Fourier operator
OpA = Fourier(mask_fromfile.mask[None])
# set device for operators
OpA_m.to(device)

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

# set device for network
if deep_decoder.device == torch.device("cpu"):
    deep_decoder = deep_decoder.to(device)
assert gpu_avail and deep_decoder.device == device, "for some reason DeepDecoder is on %s even though gpu avail %s"%(deep_decoder.device, gpu_avail)

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

# JFA's local dir
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val   = os.path.join(config.DATA_PATH, "val")
# load sample that DeepDecoder has been trained on
sample = torch.load( os.path.join(dir_train,"sample_00000.pt") )
from operators import to_complex
# go from real to complex valued sample - set imag part to zero
sample = to_complex(sample[None]).to(device)

# simulate measurements by applying the Fourier transform
measurement = OpA(sample)
measurement = measurement[None].to(device)
# load the adversarial noise (measurement adv noise)
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}_k{dim_channels}_nc{num_channels}_{architecture}".format(
    lr               = init_lr,
    gamma            = train_params["scheduler_params"]["gamma"],
    sampling_pattern = "%s_sr%.2f"%(sp_type, sampling_rate),
    dim_channels     = dim_channels,
    num_channels     = num_channels,
    architecture     = archkey,
)
adv_noise = torch.load(os.path.join(os.getcwd(), "adv_attack_dd", "adv_noise_dd_%s.pt"%(fn_suffix) ) )
# compute perturbed measurement
perturbed_measurement = measurement + adv_noise

# load same init weights as was used to find the reconstruction 
# that was the initial condition of the adversarial attack
param_dir = os.path.join(config.SCRATCH_PATH, "DeepDecoder")
deep_decoder.load_state_dict(torch.load( os.path.join(param_dir, "DeepDecoder_init_weights_%s.pt"%(fn_suffix) ) ) )

# Deep decoder input
# load ddinput
ddinput = torch.load(os.path.join(param_dir, "ddinput_%s.pt"%(fn_suffix)) )
ddinput = ddinput.to(device)

# optimizer setup
optimizer = torch.optim.Adam
scheduler = torch.optim.lr_scheduler.StepLR
optimizer = optimizer(deep_decoder.parameters(), **train_params["optimizer_params"])
scheduler = scheduler(optimizer, **train_params["scheduler_params"])

# log setup
import pandas as pd
logging = pd.DataFrame(
    columns=["loss", "lr", "psnr", "ssim"]
)
# progressbar setup
from tqdm import tqdm
progress_bar = tqdm(
    desc="Train DeepDecoder ",
    total=train_params["num_epochs"],
)
from matplotlib import pyplot as plt
num_save = train_params["num_epochs"] // train_params["save_epochs"]
fig, axs = plt.subplots(2,num_save,figsize=(5*num_save,10) )

# function that returns img of sample and the reconstructed image
from dip_utils import get_img_rec#, center_scale_01

# training loop
isave = 0
# magnitude of added gaussian noise during training
sigma_p = 1/50
for epoch in range(train_params["num_epochs"]): 
    deep_decoder.train()  # make sure we are in train mode
    optimizer.zero_grad()
    # add gaussian noise to DeepDecoder input according to Ulyanov et al 2020
    additive_noise = sigma_p*torch.randn(ddinput.shape).to(device)
    model_input    = ddinput + additive_noise
    # get img = Re(sample), img_rec = Re(pred_img), pred_img = G(ddinput, theta)
    img, img_rec, pred_img = get_img_rec(sample, model_input, model = deep_decoder)
    # pred = A G(ddinput, theta)
    pred = OpA(pred_img)
    loss = loss_func(pred, perturbed_measurement)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # compute logging metrics, first prepare predicted image
    deep_decoder.eval()
    with torch.no_grad():
        # compute prediction 
        
        img, img_rec_eval, pred_img_eval = get_img_rec(sample, ddinput, model = deep_decoder)
        # Reconstruction error
        rec_err = (sample - pred_img_eval).norm(p=2)
        # compute psnr and ssim where reconstruction is clipped between 0 and 1
        ssim_pred = ssim( img[None,None], img_rec_eval[None,None].clamp(0,1) )
        psnr_pred = psnr( img[None,None], img_rec_eval[None,None].clamp(0,1) )
    # append to log
    app_log = pd.DataFrame( 
        {
            "loss" : loss.item(), 
            "lr"   : scheduler.get_last_lr()[0],
            "psnr" : psnr_pred,
            "ssim" : ssim_pred,
        }, 
        index = [0] )
    logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
    
    # update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix(
        **deep_decoder._add_to_progress_bar({"loss": loss.item()})
    )
    if epoch % train_params["save_epochs"] == 0 or epoch == train_params["num_epochs"] - 1:
        print("Saving parameters of models and plotting evolution")
        ###### Save parameters of DeepDecoder model
        path = train_params["save_path"]
        if epoch < train_params["num_epochs"] - 1:
            torch.save(deep_decoder.state_dict(), path + "/DeepDecoder_adv_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) )
            ###### Plot evolution of training process #######
            cmap = "Greys_r"
            axs[0,isave].imshow(img_rec, cmap=cmap)
            #axs[0,isave].set_title("Epoch %i"%epoch)
            axs[1,isave].imshow( (img - img_rec)**2 , cmap=cmap)
            axs[0,isave].set_axis_off(); axs[1,isave].set_axis_off()
            isave += 1       
        else:
            torch.save(deep_decoder.state_dict(), path + "/DeepDecoder_adv_{suffix}_last.pt".format(suffix = fn_suffix) )
        
# remove whitespace and plot tighter
fig.tight_layout()
save_path_dd_adv_rec = os.path.join(config.RESULTS_PATH, "..", "plots", "adversarial_plots", "DeepDecoder")
fig.savefig(os.path.join(save_path_dd_adv_rec, "DeepDecoder_adv_evolution_%s_sr%.2f.png"%(sp_type, sampling_rate)), bbox_inches="tight")

# save final reconstruction
deep_decoder.eval()
img, img_rec, rec = get_img_rec(sample, ddinput, model = deep_decoder) 
# center and normalize to x_hat in [0,1]
img_rec = (img_rec - img_rec.min() )/ (img_rec.max() - img_rec.min() )
from dip_utils import plot_train_DIP
plot_train_DIP(img, img_rec, logging, save_fn = os.path.join(
    save_path_dd_adv_rec,
    "DeepDecoder_adv_train_metrics_%s_sr%.2f.png"%(sp_type, sampling_rate),
))
