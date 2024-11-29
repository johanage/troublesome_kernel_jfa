
# ---------------- Description ---------------------------------------------------
#  This is a testscript for using untrained NNs on methane flux inversion
#  from the Lagoon-pingo site near Longearbyen on Svalbard.
# --------------------------------------------------------------------------
# general imports
import os, sys
import matplotlib as mpl
import torch, numpy as np
from torchvision.utils import save_image
from piq import psnr, ssim
# local imports
from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import DeepDecoder
from functools import partial
from matplotlib import pyplot as plt
from math import prod

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

# ----- network configuration -----
num_channels = 5
dim_channels = 128
archdict = {
    # inspired by UNet's hourglass structure
    "unetinsp1" : [128, 64, 32, 64, 128], 
    "unetinsp2" : [128, 128, 64, 64, 128],
    # just a flat structure
    "flat"      : [dim_channels]*num_channels,
}
archkey = "flat"
deep_decoder_params = {
    "output_channels" : 1,                # 1 : grayscale, 2 : complex grayscale, 3 : RGB
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

if deep_decoder.device == torch.device("cpu"):
    deep_decoder = deep_decoder.to(device)
assert gpu_avail and deep_decoder.device == device, "for some reason DeepDecoder is on %s even though gpu avail %s"%(deep_decoder.device, gpu_avail)

# ----- training configuration -----
def loss_func(pred, tar):
    residual = pred - tar
    # convert nans to zeros
    residual = torch.nan_to_num(residual)
    return residual.pow(2).sum().pow(0.5) #/ pred.shape[0]#prod(pred.shape)

# set training parameters
num_epochs = 20 # sampling rate experiment DeepDecoder epoch nr
init_lr = 5e-3
train_params = {
    "num_epochs": num_epochs,
    "batch_size": 1,
    "loss_func": loss_func,
    "save_path": os.path.join(config.SCRATCH_PATH,"DeepDecoder"),
    "save_epochs": 2,
    "optimizer_params": {"lr": init_lr, "eps": 1e-8, "weight_decay": 0},
    "scheduler_params": {
        "step_size": 100, 
        "gamma"    : 0.98
    },
    "acc_steps": 1,
}

# ------ construct network and train -----
fn_suffix = "lr_{lr}_gamma_{gamma}_sp_{sampling_pattern}_k{dim_channels}_nc{num_channels}_{architecture}{additional}".format(
    lr               = init_lr, 
    gamma            = train_params["scheduler_params"]["gamma"],
    sampling_pattern = "drone",
    dim_channels     = dim_channels,
    num_channels     = num_channels,
    architecture     = archkey,
    additional       = "_lagoonpingo", # if multilevel sampling pattern with param a=1 is used
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

# TODO rewrite the forward model to work on tensors
def forward_model(
  pos               : torch.Tensor,
  rates             : torch.Tensor,
  source_loc        : torch.Tensor,
  particle_lifetime : float,
  coeff_diffusion   : float,
  windspeed         : torch.Tensor,
  device            : torch.device = device,
) -> torch.Tensor:
  """
  Extension of model from Vergassola et al. 2007 Infotaxis paper.
  
  Based on the stationary advection-diffusion equation (ADE)

    0 = D d^2u(x)/dx - v_a \dot u(x) - u(x)/tau + sum_i rate_i * dirac(x - x_{s_i})
  
  where \dot is the dot product, * is the symbol used for multiplication and
  - x in R^3 is the spatial coordinate
  - t in R_>=0 is the temporal coordinate
  - u in R is the scalar field (in this case the concentration with [conc.] = 1/m^3 in 3D)
  - D in R is the diffusion coefficient, [D] = m^2/s - NOTE that the diffusion coeff. not only represents the moleculare diffusion but also diffusion due to turbulence
  - v_a in R^3 is the advection velocity in 3D, [v_a] = m/s
  - tau in R is the lifetime of a particle at (x)
  - rate in R^N is the source rates at source locations x_{s_i} in R^3 (for i'th source)
  - dirac is the dirac delta function

  Inputs:
   - pos is the position at which we want to evaluate the concentration, shape = (3, Npos) where Npos is the number of positions we want to evaluate the concentration
   - rates are the source rates = (N_s) where N_s is the number of sources
   - particle_lifetime is the lifetime of particles at (x,t)
   - coeff_diffusion is the diffusion coefficient
   - windspeed is the windspeed in 3D, shape windspeed = (3)
   - source_loc is the source locations, shape source_loc = (3,N_s) where N_s is the number of sources and location in R^3

  Out:
   - concentration in R at pos (scalar value) 
  """
  # assert all inputs at correct device
  assert pos.device        == device
  assert rates.device      == device
  assert source_loc.device == device
  assert windspeed.device  == device
  # check that source parameters have same length
  assert source_loc.shape[-1] == rates.shape[-1]

  # common variables
  shape = (3, pos.shape[-1], source_loc.shape[-1],)
  nanmat = torch.ones(shape).to(device)*torch.nan

  # Fill the source locations correctly
  sltemp = nanmat.clone()
  for i in range(nanmat.shape[1]): sltemp[:,i] = source_loc
  source_loc = sltemp; del sltemp

  # Fill the measurement positions correctly
  postemp = nanmat.clone()
  for i in range(nanmat.shape[-1]): postemp[:,:,i] = pos
  pos = postemp; del postemp
  # compute the l2norm of r-r0
  l2pos_sourceloc = torch.norm(pos - source_loc,dim=0, p=2)

  # First term
  pi = 3.141592653589793
  #term1 = rates / ( 4 * pi * coeff_diffusion * l2pos_sourceloc )
  logterm1 = torch.log(rates) - torch.log( 4 * pi * coeff_diffusion * l2pos_sourceloc )

  # Driftterm
  # NOTE the shapes must be pos = (3, Npos), souce_loc = (3, Ns) and windspeed = (3)
  # e^(- (r - r0) dot V  )
  rVdotprod = torch.einsum("ijk,i->jk", pos - source_loc, windspeed )
  #term2 = torch.exp( rVdotprod/ 2 / coeff_diffusion )
  logterm2 = rVdotprod / 2 / coeff_diffusion

  # compute lambda the characteristic transport distance
  V = torch.norm(windspeed, p=2)
  difflength = coeff_diffusion * particle_lifetime
  diff_vs_drift = V**2 * particle_lifetime / 4 / coeff_diffusion
  charlength = torch.sqrt( difflength / (1 + diff_vs_drift) )

  # Exponential from source term (diffusion)
  #term3 = torch.exp( - l2pos_sourceloc / charlength)
  logterm3 = -l2pos_sourceloc / charlength

  # multiply to get concentration
  #conc_sources = term1 * term2 * term3
  conc_sources = torch.exp( logterm1 + logterm2 + logterm3)
  concentration = conc_sources.sum(dim=-1)
  return concentration.to(torch.float)


# Get methane measuremnet example 
measurement_imgformat = torch.load(os.path.join(config.DATA_PATH_LAGOON, "ch4_f19_256.pt"),weights_only=True)
#measmask              = ~measurement_imgformat.isnan() 
measmask              = torch.ones(measurement_imgformat.shape).type(torch.bool) 
measurement           = measurement_imgformat[measmask]
measurement           = measurement.type(torch.float).to(device)
# TODO determine if to use the whole domain or only explicit source locations
sourcemask     = torch.load(os.path.join(config.DATA_PATH_LAGOON, "sourcemask_256.pt"),weights_only=True)
surface        = torch.load(os.path.join(config.DATA_PATH_LAGOON, "surface3D_256.pt"),weights_only=True)
coords_measpos = torch.load(os.path.join(config.DATA_PATH_LAGOON, "coords_domain_256.pt"),weights_only=True)
sourcepos = surface[:,sourcemask.ravel()] 
measpos   = coords_measpos.type(torch.float)#[:,measmask.ravel()]
# Move all relevant tensors to GPU
sourcemask = sourcemask.type(torch.bool).to(device)
surface = surface.type(torch.float).to(device)
coords_measpos = coords_measpos.type(torch.float).to(device)
sourcepos = sourcepos.type(torch.float).to(device)
measpos = measpos.to(device)
windspeed = torch.tensor([2.5,-2.5,0]).type(torch.float).to(device)
# simplify forward model function
OpA = partial(
  forward_model,
  pos               = measpos,
  #rates -> comes from the DeepDecoder
  source_loc        = sourcepos,
  particle_lifetime = 1,
  coeff_diffusion   = 5,
  windspeed         = windspeed,
)
# test forward model by making a prediction on all positions
# -> use testrates as solution to synthetic measurements from concentrations OpA(rates)
init_rate = 1e5
testrates_img = init_rate*torch.ones(sourcemask.shape).to(device)
testrates_img[sourcemask] = 0
testrates = testrates_img[sourcemask].to(device)
measurement = OpA(rates=testrates)

# plot, save and export to confirm correctness of results
#testpred_img = torch.nan*torch.ones(measmask.shape).to(device)
#testpred_img[measmask] = testpred.type(torch.float)

# NOTE this is only relevant for simulated measurements
#meas_noise_std = 0.08 * measurement.norm(p=2) / np.sqrt(np.prod( measurement.shape[-2:] ) )
#meas_noise = meas_noise_std * torch.randn_like(measurement)
#measurement += meas_noise 
#print(" l2-norm of Gaussaian noise : ", meas_noise.norm(p=2)/ OpA(sample).norm(p=2) ) 

# ---------- Deep decoder input -------------------
# height and width of input vector
# deep_decoder.nsacles = number of layers
in_hw = 256//2**(deep_decoder.nscales-1) # of nlayers = 5 then in_hw = 256//2**4 = 256//16 = 16
input_range = [0,10]
save_ddinput = True
# save/load ddinput in case needed for adv. noise study
if save_ddinput:
    # torch.rand : u ~ U([0,1] so for input_range=[a,b] we get a + b * u ~ U([a,b])
    # input dim = (1, 128, 16, 16) if flat arch with layer size 128 and nlayer = 5 making input height and width 256//2**4 = 16
    ddinput = input_range[0] + input_range[1] * torch.rand((1, deep_decoder_params["channels_up"][0],in_hw, in_hw))
    torch.save(ddinput, os.path.join(train_params["save_path"], "ddinput_%s.pt"%(fn_suffix)) )
else:
    ddinput = torch.load(os.path.join(train_params["save_path"], "ddinput_%s.pt"%(fn_suffix)) )
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

# init figure to plot evolution
from matplotlib import pyplot as plt
num_save = train_params["num_epochs"] // train_params["save_epochs"]
fig, axs = plt.subplots(2,num_save,figsize=(5*num_save,10) )

# magnitude of added gaussian noise during training - noise-based regularisation
sigma_p = 1/50
reconstruct = True
#TODO figure out how to scale to physical quantities
if reconstruct:
    for epoch in range(train_params["num_epochs"]): 
        deep_decoder.train()  # make sure we are in train mode
        optimizer.zero_grad()
        # noise-based regularisation: add gaussian noise to DeepDecoder input according to Ulyanov et al 2020
        additive_noise = sigma_p*torch.randn(ddinput.shape).to(device)
        model_input    = ddinput + additive_noise
        # pred_img = G(model_input, theta), model_input = ddinput + noise
        pred_img = deep_decoder.forward(model_input)
        # pred = f ( G(ddinput, theta) )
        pred = OpA(rates= init_rate * pred_img[0,0][sourcemask])
        loss = loss_func(
            pred[measmask.ravel()], 
            measurement[measmask.ravel()],
        )
        breakpoint()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # compute logging metrics, first prepare predicted image
        deep_decoder.eval()
        with torch.no_grad():
            # get surface flux reconstruction
            pred_img_eval = deep_decoder.forward(ddinput).cpu()
            # Reconstruction error
            ref_img = testrates_img[None,None].cpu()
            refscale = ref_img.max()
            rec_err = (ref_img - refscale*pred_img_eval).norm(p=2)
            # SSIM
            ssim_pred = ssim(torch.nan_to_num(ref_img)/refscale, torch.nan_to_num(pred_img_eval)/refscale)
            # PSNR
            psnr_pred = psnr(torch.nan_to_num(ref_img)/refscale, torch.nan_to_num(pred_img_eval)/refscale)
            # compute the difference between the complex image in train mode vs. eval mode
            rel_eval_diff = (torch.log( torch.nan_to_num(pred_img.detach().cpu() - pred_img_eval).norm(p=2)/ torch.nan_to_num(pred_img).norm(p=2) ) / torch.log(torch.tensor(10)) )
        
        # append to log
        app_log = pd.DataFrame(
            {
                "loss"          : loss.item(), 
                "rel_eval_diff" : rel_eval_diff.item(),
                "psnr"          : psnr_pred.item(),
                "ssim"          : ssim_pred.item(),
                "rec_err"       : rec_err.item(),
                "rel_rec_err"   : (rec_err.item() / torch.nan_to_num(ref_img).norm(p=2) ).item(),
                "lr"            : scheduler.get_last_lr()[0],
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
            img_rec = pred_img_eval[0,0]
            img     = ref_img[0,0]
            
            if epoch < train_params["num_epochs"] - 1:
                # save rec. image as .pt
                torch.save(pred_img.detach().cpu(), os.path.join(path, "DeepDecoder_nojit_rec_{suffix}_epoch{epoch}.pt".format(
                    suffix = fn_suffix,
                    epoch  = epoch,
                )))
                
                # save the deep_decoder.parameters for each num_epochs
                torch.save(deep_decoder.state_dict(), os.path.join(path,"DeepDecoder_nojit_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) ) )
            else:
                # save last rec img
                torch.save(pred_img.detach().cpu(), os.path.join(path, "DeepDecoder_nojit_rec_{suffix}_last.pt".format(
                    suffix = fn_suffix,
                    epoch  = epoch,
                )))
                # save rec. image as .png
                save_image(img_rec.detach().cpu(), os.path.join(config.PLOT_PATH, "DeepDecoder", "sample_rate_experiment", "DeepDecoder_nojit_rec_{suffix}_last.png".format(
                    suffix = fn_suffix,
                )))
                # save last deep_decoder.params
                torch.save(deep_decoder.state_dict(), os.path.join(path,"DeepDecoder_nojit_{suffix}_last.pt".format(suffix = fn_suffix) ) )

# save/load the logging table to pickle
save_logging = True
if save_logging:
    logging.to_pickle(os.path.join(config.RESULTS_PATH, "DeepDecoder", "test-DeepDecoder_nojit_logging_{suffix}.pkl".format(suffix=fn_suffix) ))
else:
    logging = pd.read_pickle(os.path.join(config.RESULTS_PATH, "DeepDecoder", "test-DeepDecoder_nojit_logging_{suffix}.pkl".format(suffix=fn_suffix) ))


# plot evolution
for epoch in range(train_params["num_epochs"]):
    if epoch % train_params["save_epochs"] == 0:
            # load parameters at epcoch 
            deep_decoder.load_state_dict( torch.load(os.path.join(path,"DeepDecoder_nojit_{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch) ) ) )
            deep_decoder.eval()
            # reconstruct image at epoch
            #pred_img_eval = deep_decoder.forward(model_input).cpu()
            pred_img_eval = deep_decoder.forward(ddinput).cpu().detach()
            # add image eval metrics as text to residual row
            #rec_psnr = psnr(img_rec[None, None].clamp(0,1), img.cpu())
            #rec_ssim = ssim(img_rec[None, None].clamp(0,1), img.cpu())
            ###### Plot evolution of training process #######
            cmap = "Greys_r"
            # turn of scale at epoch 0 so you see the prior better
            if epoch == 0:
                axs[0,isave].imshow(img_rec, cmap=cmap, origin="lower", interpolation="none")
            else:
                axs[0,isave].imshow(img_rec, cmap=cmap, origin="lower", interpolation="none", vmin=0, vmax=1)
            save_image(img_rec.T, os.path.join(config.PLOT_PATH, "DeepDecoder", "evolution", "DeepDecoder_nojit_epoch{epoch}_{suffix}.png".format(
                suffix = fn_suffix,
                epoch  = epoch,
            )) )
            #axs[0,isave].set_title("Epoch %i"%epoch)
            axs[1,isave].imshow((img - img_rec).abs(), origin="lower", interpolation="none", cmap=cmap, vmin=0, vmax=.2)
            axs[0,isave].set_axis_off(); axs[1,isave].set_axis_off()
            #axs[1,isave].text(x = 5,y = 60, s = "PSNR : %.1f \nSSIM : %.2f"%(rec_psnr, rec_ssim), fontsize = 36, color="white")
            isave += 1
        
# plot evolution of training
fig.tight_layout()
fig_savepath = os.path.join(config.PLOT_PATH, "DeepDecoder", "sample_rate_experiment")
fig.savefig(os.path.join(fig_savepath, "DeepDecoder_nojit_evolution_{suffix}.png".format(suffix=fn_suffix)), bbox_inches="tight")

# save final reconstruction
from dip_utils import plot_train_DIP
with torch.no_grad():
    deep_decoder.eval()
    out_dd_eval = deep_decoder(ddinput)
    img_dd_eval = out_dd_eval.norm(p=2, dim=(0,1))
    save_image(img_dd_eval.T, os.path.join(
        fig_savepath, 
        "DeepDecoder_nojit_rec_{suffix}_last.png".format(
            suffix = fn_suffix,
    )))
    plot_train_DIP(img.cpu(), img_rec, logging, save_fn = os.path.join(fig_savepath, "DeepDecoder_nojit_train_metrics_{suffix}.png".format(suffix=fn_suffix)) )
