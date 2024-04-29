# python libs imports
import os
import matplotlib as mpl
import torch
from torchvision.utils import save_image
from piq import psnr, ssim
# local imports
from data_management import IPDataset, SimulateMeasurements, ToComplex
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
"""
mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
"""
sr_list = [0.03, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.23, 0.25]
sampling_rate = sr_list[-1]
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    #path = os.path.join(config.SP_PATH, "circle"),
    #filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate)
    # -------- a=2 Samplig patterns --------------------
    path = config.SP_PATH,
    filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png",
)

# Fourier matrix
OpA_m = Fourier_m(mask_fromfile.mask[None])
# Fourier operator
OpA = Fourier(mask_fromfile.mask[None])
# set device for operators
OpA_m.to(device)

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
# ------ construct network with chosen architecture -----
unet = unet(**unet_params)

if unet.device == torch.device("cpu"):
    unet = unet.to(device)
assert gpu_avail and unet.device == device, "for some reason unet is on %s even though gpu avail %s"%(unet.device, gpu_avail)

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")
def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

# set training parameters
num_epochs = 20000
init_lr = 1e-4
savedir = os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "a2", "adv_rec")
if not os.path.exists(savedir):
    os.makedirs(savedir)
train_params = {
    "num_epochs": num_epochs,
    "batch_size": 1,
    "loss_func": loss_func,
    "save_path": savedir,
    "save_epochs": num_epochs//10,
    "optimizer_params": {"lr": init_lr, "eps": 1e-8, "weight_decay": 0},
    "scheduler_params": {"step_size": 100, "gamma": 0.99},
    "acc_steps": 1,
}

# JFA's local dir
dir_train = os.path.join(config.DATA_PATH, "train")
dir_val   = os.path.join(config.DATA_PATH, "val")
from operators import to_complex
# go from real to complex valued sample - set imag part to zero
# load sample that DIP has been trained on
sample = torch.load( os.path.join(dir_val,"sample_00021_text.pt") )
sample = to_complex(sample[None, None]).to(device)

# simulate measurements by applying the Fourier transform
measurement = OpA(sample)
measurement = measurement.to(device)
# load the adversarial noise (measurement adv noise)
#adv_noise = torch.load(os.path.join(os.getcwd(), "adv_attack_dip", "adv_noise_dip_x_%s_sr%.2f.pt"%(sp_type, sampling_rate)) )
noiserel = 8e-2
perturbed_measurement = torch.load(os.path.join(
    os.getcwd(), 
    "adv_attack_dip", 
    "adv_example_noiserel%.2f_DIP_UNet_nojit_lr_0.0001_gamma_0.99_step_100_sp_%s_sr%.2f_a2_last.pt"%(noiserel, sp_type, sampling_rate)
) )
# compute perturbed measurement
#perturbed_measurement = measurement + adv_noise

# temp dir with network weights and z_tilde
#param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "DIP")
param_dir = os.path.join(config.RESULTS_PATH_KADINGIR, "DIP", "a2")

# init noise vector (as input to the model) z ~ U(0,1/10) - DIP paper SR setting
z_tilde = torch.load(os.path.join(param_dir, "z_tilde_%s_%.2f.pt"%(sp_type, sampling_rate)) )
z_tilde = z_tilde.to(device)

# load model weights
file_param = "DIP_UNet_nojit_lr_0.0001_gamma_0.99_step_100_sp_%s_sr%.2f_a2_last.pt"%(sp_type, sampling_rate)
#params_loaded = torch.load( os.path.join(param_dir, file_param) )
params_loaded = torch.load( os.path.join(param_dir, "DIP_UNet_init_weights_%s_%.2f.pt"%(sp_type, sampling_rate)) )
unet.load_state_dict(params_loaded)
with torch.no_grad():
    unet.eval()
    save_image(unet.forward(z_tilde).cpu().norm(p=2, dim=(0,1)), os.path.join(param_dir, "xhat0_before_adv_rec.png") )

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
from dip_utils import get_img_rec#, center_scale_01

###### Save parameters of DIP model
path = train_params["save_path"]
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}_noiserel{noiserel}".format(
    lr               = init_lr, 
    gamma            = train_params["scheduler_params"]["gamma"],
    sampling_pattern = "%s_sr%.2f"%(sp_type, sampling_rate),
    noiserel         = "%i"%(int(100*noiserel)),
)

# -------- training loop ---------------------------------
isave = 0
# magnitude of added gaussian noise during training
sigma_p = 1/30
for epoch in range(train_params["num_epochs"]): 
    unet.train()  # make sure we are in train mode
    optimizer.zero_grad()
    # add gaussian noise to DIP input according to Ulyanov et al 2020
    additive_noise = sigma_p*torch.randn(z_tilde.shape).to(device)
    model_input    = z_tilde + additive_noise
    # get img = Re(sample), img_rec = Re(pred_img), pred_img = G(z_tilde, theta)
    img, img_rec, pred_img = get_img_rec(sample, model_input, model = unet)
    # pred = A G(z_tilde, theta)
    pred = OpA(pred_img)
    loss = loss_func(pred, perturbed_measurement)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # compute logging metrics, first prepare predicted image
    with torch.no_grad():
        unet.eval()
        # compute prediction 
        #img, img_rec_eval, pred_img_eval = get_img_rec(sample, z_tilde, model = unet)
        img = sample.norm(p=2, dim=(0,1)).cpu()
        pred_img_eval = unet.forward(z_tilde).cpu()
        img_rec_eval = pred_img_eval.norm(p=2, dim=(0,1))
        # Reconstruction error
        rec_err = (sample.cpu() - pred_img_eval).norm(p=2)
        # compute psnr and ssim where reconstruction is clipped between 0 and 1
        ssim_pred = ssim( img[None,None], img_rec_eval[None,None].clamp(0,1) )
        psnr_pred = psnr( img[None,None], img_rec_eval[None,None].clamp(0,1) )
    # append to log
    app_log = pd.DataFrame( 
        {
            "loss"        : loss.item(), 
            "lr"          : scheduler.get_last_lr()[0],
            "psnr"        : psnr_pred,
            "ssim"        : ssim_pred,
            "rec_err"     : rec_err.item(),
            "rel_rec_err" : (rec_err.item() / sample.norm(p=2) ).item(),
        }, 
        index = [0] )
    logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
    
    # update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix(
        **unet._add_to_progress_bar({"loss": loss.item()})
    )
    if epoch % train_params["save_epochs"] == 0 or epoch == train_params["num_epochs"] - 1:
        print("Saving parameters of models and plotting evolution")
        
        if epoch < train_params["num_epochs"] - 1:
            torch.save(unet.state_dict(), path + "/DIP_UNet_adv_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) )
            save_image(img_rec_eval.cpu(), path + "/DIP_UNet_adv_rec_{suffix}_epoch{epoch}.png".format(suffix = fn_suffix, epoch=epoch) )
            ###### Plot evolution of training process #######
            cmap = "Greys_r"
            axs[0,isave].imshow(img_rec, cmap=cmap)
            #axs[0,isave].set_title("Epoch %i"%epoch)
            axs[1,isave].imshow(.5*torch.log( (img - img_rec)**2), cmap=cmap)
            axs[0,isave].set_axis_off(); axs[1,isave].set_axis_off()
            isave += 1       
        else:
            torch.save(unet.state_dict(), path + "/DIP_UNet_adv_{suffix}_last.pt".format(suffix = fn_suffix) )
            # save last rec img
            torch.save(pred_img_eval.detach().cpu(), os.path.join(path, "DIP_nojit_rec_{suffix}_last.pt".format(
                suffix = fn_suffix,
            )))
            save_image(img_rec_eval.cpu(), path + "/DIP_UNet_adv_rec_{suffix}_last.png".format(suffix = fn_suffix) )

# save the logging table to pickle
save_logging = True
if save_logging:
    logging.to_pickle(os.path.join(path, "DIP_UNet_nojit_logging_{suffix}.pkl".format(suffix = fn_suffix)))
# load logging
else:
    logging = pd.read_pickle(os.path.join(path, "DIP_UNet_nojit_logging_{suffix}.pkl".format(suffix = fn_suffix)))
   
# remove whitespace and plot tighter
fig.tight_layout()
save_path_dip_adv_rec = os.path.join(config.PLOT_PATH, "adversarial_plots", "DIP", "noiserel%i"%( int(noiserel*100) ) )
if not os.path.exists(save_path_dip_adv_rec):
    os.makedirs(save_path_dip_adv_rec)

fig.savefig(os.path.join(save_path_dip_adv_rec, "DIP_adv_evolution_%s_sr%.2f.png"%(sp_type, sampling_rate)), bbox_inches="tight")

# save final reconstruction
unet.eval()
img, img_rec, rec = get_img_rec(sample, z_tilde, model = unet) 
# center and normalize to x_hat in [0,1]
img_rec = (img_rec - img_rec.min() )/ (img_rec.max() - img_rec.min() )
from dip_utils import plot_train_DIP
plot_train_DIP(img.cpu(), img_rec.cpu(), logging, save_fn = os.path.join(
    save_path_dip_adv_rec,
    "DIP_adv_train_metrics_%s_sr%.2f.png"%(sp_type, sampling_rate),
))
