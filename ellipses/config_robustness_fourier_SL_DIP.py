# Robustness test configuration for
# - Supervised learning: wiht and without jitter
# - Deep image prior - serving as untrained network/ unsupervised method
import os
from typing import Tuple, Type, Callable
from functools import partial
import matplotlib as mpl
import pandas as pd, torch, torchvision, numpy as np
from piq import psnr, ssim

from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import UNet, DeepDecoder
from operators import (
    Fourier,
    Fourier_matrix as Fourier_m,
    LearnableInverterFourier,
    RadialMaskFunc,
    MaskFromFile,
    noise_gaussian,
    proj_l2_ball,
)
from find_adversarial import (
    PGD,
    PAdam,
    PAdam_DIP_x,
    untargeted_attack,
    grid_attack,
)

# ----- load data configuration -----
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
mask_fromfile = MaskFromFile(
    path = os.getcwd() + "/sampling_patterns/",
    filename = "multilevel_sampling_pattern_sr2.500000e-01_a2_r0_2_levels50.png"
)
mask = mask_fromfile.mask[None]
# Fourier matrix
OpA_m = Fourier_m(mask)
# Fourier operator
OpA = Fourier(mask)
# Inverse Fourier operator, can be set to learnable
inverter = LearnableInverterFourier(config.n, mask, learnable=False)
# set device for operators
OpA_m.to(device)
inverter.to(device)

# ----- Reconstruciton methods ------------------------------------------------
methods = pd.DataFrame(columns=["name", "info", "reconstr", "rec_config", "attacker", "net"])
methods = methods.set_index("name")
# reference noise function is set to gaussian additive noise, see operators.py for description
noise_ref = noise_gaussian

# ----- set up net attacks --------
# loss functions
mseloss = torch.nn.MSELoss(reduction="sum")
def _complexloss(reference, prediction):
    loss = mseloss(reference, prediction)
    return loss

def loss_func(pred, tar):
    return mseloss(pred, tar) / pred.shape[0]

# first step loss function DIP on xhat
from dip_utils import loss_adv_example
#l_adv = ||A xhat - y_adv||_2^2 - beta * || x - xhat||_2^2
# A = meas_op, 
loss_adv_partial = partial(loss_adv_example,  meas_op = OpA, beta = 1e-3)


# the actual reconstruction method for any supervised net
# def _reconstructNet(y, noise_rel, net):
def _reconstructNet(y, net):
    return net.forward(y)

# unsure if necessary
from dip_utils import _reconstructDIP

# attack function for any net
from typing import Dict
def _attackerNet(
    x0           : torch.Tensor,
    noise_rel    : float,
    net          : torch.nn.Module,
    yadv_init    : torch.Tensor = None,
    adv_optim    : Callable     = PAdam,
    rec_config   : dict         = None,
    batch_size   : int          = 3,
    lr_adv_optim : float        = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    -----------------------------------------------------------
    Untargeted attack function for any DNN
    
    Untargeted attacks outputs untarget examples which are measurements
    y' = y + delta that lies close to ground truth measurement y,
    i.e. ||delta|| <= epsilon of image x such that Psi(y') != x.
    Furthermore we maximize the norm ||Psi(y') - x|| with optimization
    method given in dictionary adv_param.
    -----------------------------------------------------------
    Args:
    - x0           : ground truth image
    - noise_rel    : noise level relative to the ground truth image
    - net          : DNN to use
    - yadv_init    : initial adversarial measurements
    - rec_config   : dict containing reconstruction function and necessary config
    - batch_size   : batch size
    - lr_adv_optim : learning rate for adversarial example optimization, 
                     for PGD line search is included so lr can be set to much higher than 
                     PAdam-based
    Out:
    - yadv : adversarial measurement
    - yref : refrence measuruement (measurement with gaussian noise)
    - y0   : noiseless measurment
    """
    
    # set reconstruction function
    if rec_config is None or rec_config["reconstruction_method"] == "Supervised":
        rec = lambda y: _reconstructNet(y, net)
    # this can be changed to string instead of boolean condition if more learning paradigms are added
    else:
        # setup for DIP that computes new reconstructions explicitly
        if rec_config["reconstruction_method"] in ["DIP_x", "DeepDecoder"]:
            rec   = rec_config["rec_func_adv_noise"]
            # this is computed and added to the rec_config dict in the grid attack in find_adversarial.py
            xhat0 = rec_config["xhat0"]
   
    """-------------------------------------------------------------------------------------------------------------
       After this step the reconstruction function "rec" can only take a single argument which are the measurements.
       -------------------------------------------------------------------------------------------------------------"""
    # compute noiseless measurements
    y0 = OpA(x0)
    # adv. and noisy measurements will be noiseless with noise_rel = 0
    if noise_rel == 0.0:
        return y0, y0, y0
    # compute absolute noise levels
    noise_level = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    # compute noisy measurements for reference
    yref = noise_ref(OpA(x0), noise_level)  # noisy measurements
    # attack parameters
    adv_init_fac = 3.0 * noise_level
    adv_param = {
        #"codomain_dist" : _complexloss,
        "codomain_dist" : rec_config["codomain_distance"],
        "domain_dist"   : None,
        "mixed_dist"    : None,
        "weights"       : (1.0, 1.0, 1.0),
        "optimizer"     : adv_optim,
        "projs"         : None,
        "niter"         : rec_config["niter_adv_optim"],#1000,
        "stepsize"      : 1e-3,
    }
    # compute initialization
    # adv_init_fac = k * noise_rel * ||y0||_2, default: k=3
    adv_noise_mag = adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    # perturbed measurements = measurments + C * z, z~N(0,I) z in C^m
    yadv = y0.clone().detach() + adv_noise_mag * torch.randn_like(y0)
    # set init adversarial meas example if previous lower noise level was computed
    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()
    
    # set batch size to defined in rec_config
    # - DIP methods must have batch size = 1
    if "batch_size" in list(rec_config.keys() ):
        batch_size = rec_config["batch_size"]
    for idx_batch in range(0, yadv.shape[0], batch_size):
        print(
            "Attack for samples "
            + str(list(range(idx_batch, idx_batch + batch_size)))
        )
        y0_batch = y0[idx_batch : idx_batch + batch_size, ...]
        x0_batch = x0[idx_batch : idx_batch + batch_size, ...]
        
        # set l2-ball projection
        # - centered at y0[idx_batch : idx_batch + batch_size, ...]
        # - with radius given by noise_level[idx_batch : idx_batch + batch_size, ...]
        adv_param["projs"] = [
            lambda y: proj_l2_ball(
                y,
                y0_batch,
                noise_level[idx_batch : idx_batch + batch_size, ...],
            )
        ]
        # perform untargeted attack
        if "DIP" in rec_config["reconstruction_method"] or "DeepDecoder" in rec_config["reconstruction_method"]:
            xhat0_batch = xhat0[idx_batch : idx_batch + batch_size, ...]
            # update so that the reconstructed image 
            # xhat0 is returned in rec in the first iteration
            rec = partial(rec, xhat = xhat0_batch)
            adv_param["optimizer"] = partial(adv_param["optimizer"], xhat0 = xhat0_batch)

        # compute adversarial example for batch
        yadv[idx_batch : idx_batch + batch_size, ...] = untargeted_attack(
            func      = rec,
            t_in_adv  = yadv[idx_batch : idx_batch + batch_size, ...].clone().requires_grad_(True),
            t_in_ref  = y0_batch,
            t_out_ref = x0_batch,
            **adv_param
        ).detach()

    return yadv, yref, y0

# ----- Load nets -----

# create a net and load weights from file
def _load_net(
    path       : str, 
    net        : torch.nn.Module, 
    net_params : dict,
) -> torch.nn.Module:
    """
    Loads the parameter values of net with parameters net_params from path.
    
    Args:
    - path       : path to saved network parameter values.
    - net        : network class
    - net_params : params specifying network architecture ++
    
    Out:
    - Network with parameters given by state dict loaded from path.
    """
    network = net(**net_params).to(device)
    network.load_state_dict(torch.load(path, map_location=torch.device(device)))
    # freeze network weights
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.eval()
    return network

# methods dataframe append function for each net configuration
def _append_net(
    name       : str, 
    info       : dict, 
    net        : torch.nn.Module,
    adv_optim  : Callable = PAdam,
    rec_config : dict = None,
) -> None:
    """
    Appends reconstruction method to methods dataframe defined locally above.
    
    Args:
    - name       : name of the method
    - info       : metadata of the method for logging purposes
    - net        : network loaded with pretrained parameters
    - adv_optim  : Optimizer function for adversarial noise optimization. 
    - rec_config : config with explicit reconstruction method and relevant keyword arguments
    Out: Nothing
    """
    # assume supervised if no reconstruction configuration is specified
    if rec_config is None:
        rec = partial(_reconstructNet, net=net)
        rec_config = {
            "reconstruction_method"   : "supervised",
            "reconstruction_function" : rec, 
        }
    else:
        rec = rec_config["reconstruction_function"]
        #if isinstance(rec, partial):
        #    if "net" in rec.func.__code__.co_varnames:
        #        rec = partial(rec, net = net) 
        if isinstance(rec, Callable) and not isinstance(rec, partial):
            if "net" in rec.__code__.co_varnames:
                rec = partial(rec, net=net)
    methods.loc[name] = {
        "info"       : info,
        "reconstr"   : rec,
        "rec_config" : rec_config,
        "attacker"   : lambda x0, noise_rel, yadv_init=None, rec_config=rec_config: _attackerNet(
            x0, noise_rel, net, yadv_init = yadv_init, adv_optim = adv_optim, rec_config = rec_config,
        ), 
        "net"        : net,
    }

# ----- DIP UNet configuration -----
dip_unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : None,
    "upsampling"    : "nearest",
}

# optimization config for final reconstruction
dip_nepochs             = 20000
dip_optimizer_params    = {"lr": 1e-4, "eps": 1e-8, "weight_decay": 0}
dip_f_optimizer         = lambda net_params, opt_params=dip_optimizer_params : torch.optim.Adam(net_params, **opt_params)
dip_lr_scheduler_params = {"step_size": 100, "gamma": 0.99} 
dip_f_lr_scheduler      = lambda optimizer, scheduler_params=dip_lr_scheduler_params : torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

from functools import partial
# DIP no jitter max number of epochs/iterations
_append_net(
    name = "DIP UNet no jit",
    info = {
        "name_disp"     : "DIP UNet w/o noise",
        "name_save"     : "dip_unet_nojit_3_3",
        "plt_color"     : "#b3b3b3",
        "plt_marker"    : "d",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    net = UNet(**dip_unet_params),
    adv_optim  = PAdam_DIP_x, 
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "DIP_x",
        "reconstruction_function" : partial(
            _reconstructDIP, 
            f_optimizer = dip_f_optimizer,
            f_scheduler = dip_f_lr_scheduler,
            OpA         = OpA,
            epochs      = dip_nepochs,
        ),
        "rec_func_adv_noise" : lambda y, xhat : xhat,
        "codomain_distance"  : loss_adv_partial,
        "batch_size"         : 1,
        "net_in_channels"    : dip_unet_params["in_channels"],
        "save_ztilde"        : True,
    }
)
# DIP no jitter 2/3 of max epochs/iterations
_append_net(
    name = "DIP UNet no jit 2/3 iterations",
    info = {
        "name_disp"     : "DIP UNet 2/3",
        "name_save"     : "dip_unet_nojit_2_3",
        "plt_color"     : "#5c5c5c",
        "plt_marker"    : "s",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    net = UNet(**dip_unet_params),
    adv_optim  = PAdam_DIP_x,
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "DIP_x",
        "reconstruction_function" : partial(
            _reconstructDIP,
            f_optimizer = dip_f_optimizer,
            f_scheduler = dip_f_lr_scheduler,
            OpA         = OpA,
            epochs      = 2*(dip_nepochs//3),
        ),
        "rec_func_adv_noise" : lambda y, xhat : xhat,
        "codomain_distance"  : loss_adv_partial,
        "batch_size"         : 1,
        "net_in_channels"    : dip_unet_params["in_channels"],
        "save_ztilde"        : True,
    }
)

# DIP no jitter 1/3 of max epochs/iterations
_append_net(
    name = "DIP UNet no jit 1/3 iterations",
    info = {
        "name_disp"     : "DIP UNet 1/3",
        "name_save"     : "dip_unet_nojit_1_3",
        "plt_color"     : "#000000",
        "plt_marker"    : "^",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    net = UNet(**dip_unet_params),
    adv_optim  = PAdam_DIP_x,
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "DIP_x",
        "reconstruction_function" : partial(
            _reconstructDIP,
            f_optimizer = dip_f_optimizer,
            f_scheduler = dip_f_lr_scheduler,
            OpA         = OpA,
            epochs      = dip_nepochs//3,
        ),
        "rec_func_adv_noise" : lambda y, xhat : xhat,
        "codomain_distance"  : loss_adv_partial,
        "batch_size"         : 1,
        "net_in_channels"    : dip_unet_params["in_channels"],
        "save_ztilde"        : True,
    }
)


# DIP jitter
_append_net(
    name = "DIP UNet jit",
    info = {
        "name_disp"     : "DIP UNet w/ noise",
        "name_save"     : "dip_unet_jit_plus_jit_meas",
        "plt_color"     : "#fc7272",
        "plt_marker"    : "+",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    net = UNet(**dip_unet_params),
    adv_optim  = PAdam_DIP_x, 
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "DIP_x",
        "reconstruction_function" : partial(
            _reconstructDIP, 
            f_optimizer  = dip_f_optimizer,
            f_scheduler  = dip_f_lr_scheduler,
            OpA          = OpA,
            epochs       = dip_nepochs,
            jitter_level = 0.1,
        ),
        "rec_func_adv_noise" : lambda y, xhat : xhat,
        "codomain_distance"  : loss_adv_partial,
        "batch_size"         : 1,
        "net_in_channels"    : dip_unet_params["in_channels"],
        "save_ztilde"        : True,
    }
)
# ----- Supervised UNet configuration -----
supervised_unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : inverter,
    "upsampling"    : "nearest"
}

# without jittering
_append_net(
    "Supervised UNet no jit",
    {
        "name_disp"     : "Supervised UNet w/o noise",
        "name_save"     : "unet_no_jit",
        "plt_color"     : "#b0b0b0",
        "plt_marker"    : "d",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    _load_net(
        #f"{config.SCRATCH_PATH}/supervised/circle_sr0.25/Fourier_UNet_no_jitter_brain_fastmri_256/train_phase_1/"
        f"{config.RESULTS_PATH_KADINGIR}/supervised/circle_sr0.25_a2/Fourier_UNet_no_jitter_brain_fastmri_256/train_phase_1/"
        + "model_weights.pt",
        UNet,
        supervised_unet_params,
    ),
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "Supervised",
        "reconstruction_function" : _reconstructNet,
        "rec_func_adv_noise"      : _reconstructNet,
        "codomain_distance"       : _complexloss,

    }
)

# ----------- with jittering -------------------------------------------------------------------------------------------------
# jittering level p=100
_append_net(
    "Supervised UNet jit very high noise",
    {
        "name_disp"     : "Supervised UNet w/ high noise",
        "name_save"     : "unet_jit_very_high_noise",
        "plt_color"     : "#000000",
        "plt_marker"    : "^",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    _load_net(
        #f"{config.RESULTS_PATH}/supervised/circ_sr0.25/Fourier_UNet_jitter_brain_fastmri_256eta_100.000_train_phase_2/"
        f"{config.RESULTS_PATH_KADINGIR}/supervised/circle_sr0.25_a2/Fourier_UNet_jitter_brain_fastmri_256/eta_100.000_train_phase_1/"
        + "model_weights.pt",
        UNet,
        supervised_unet_params,
    ),
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "Supervised",
        "reconstruction_function" : _reconstructNet,
        "rec_func_adv_noise"      : _reconstructNet,
        "codomain_distance"       : _complexloss,

    }
)

# jittering level p=10
_append_net(
    "Supervised UNet jit",
    {
        "name_disp"     : "Supervised UNet w/ moderate noise",
        "name_save"     : "unet_jit_high_noise",
        "plt_color"     : "#5c5c5c",
        "plt_marker"    : "s",
        "plt_linestyle" : "-",
        "plt_linewidth" : 2.75,
    },
    _load_net(
        #f"{config.RESULTS_PATH}/supervised/circ_sr0.25/Fourier_UNet_jitter_brain_fastmri_256eta_10.000_train_phase_2/"
        f"{config.RESULTS_PATH_KADINGIR}/supervised/circle_sr0.25_a2/Fourier_UNet_jitter_brain_fastmri_256/eta_10.000_train_phase_1/"
        + "model_weights.pt",
        UNet,
        supervised_unet_params,
    ),
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "Supervised",
        "reconstruction_function" : _reconstructNet,
        "rec_func_adv_noise"      : _reconstructNet,
        "codomain_distance"       : _complexloss,

    }
)

# jittering level p=1/10
_append_net(
    "Supervised UNet jit low noise",
    {
        "name_disp"     : "Supervised UNet w/ low noise",
        "name_save"     : "unet_jit_low_noise",
        "plt_color"     : "#7e7e7e",
        "plt_marker"    : "x",
        "plt_linestyle" : "--",
        "plt_linewidth" : 2.75,
    },
    _load_net(
        #f"{config.RESULTS_PATH}/supervised/circ_sr0.25/Fourier_UNet_jitter_brain_fastmri_256eta_0.100_train_phase_2/"
        #f"{config.SCRATCH_PATH}/supervised/circ_sr0.25/Fourier_UNet_jitter_brain_fastmri_256eta_0.100_train_phase_2/"
        f"{config.RESULTS_PATH_KADINGIR}/supervised/circle_sr0.25_a2/Fourier_UNet_jitter_brain_fastmri_256/eta_0.100_train_phase_1/"
        + "model_weights.pt",
        UNet,
        supervised_unet_params,
    ),
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "Supervised",
        "reconstruction_function" : _reconstructNet,
        "rec_func_adv_noise"      : _reconstructNet,
        "codomain_distance"       : _complexloss,

    }
)

# jittering mod: phase 1: p=10, phase 2: p=0 
_append_net(
    "Supervised UNet jit mod",
    {
        "name_disp"     : "Supervised UNet w/ noise mod",
        "name_save"     : "unet_jit_mod",
        "plt_color"     : "#fc7272",
        "plt_marker"    : "+",
        "plt_linestyle" : "--",
        "plt_linewidth" : 2.75,
    },
    _load_net(
        #f"{config.RESULTS_PATH}/supervised/circ_sr0.25/Fourier_UNet_jitter_mod_brain_fastmri_256eta_10.000_train_phase_2/"
        f"{config.RESULTS_PATH_KADINGIR}/supervised/circle_sr0.25_a2/Fourier_UNet_jitter_mod_brain_fastmri_256/eta_0.100_train_phase_1/"
        + "model_weights.pt",
        UNet,
        supervised_unet_params,
    ),
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "Supervised",
        "reconstruction_function" : _reconstructNet,
        "rec_func_adv_noise"      : _reconstructNet,
        "codomain_distance"       : _complexloss,

    }
)

# ---------------- DeepDecoder ----------------
#                  Adv. attack
# --------- Network confiuguration
num_channels = 5
dim_channels = 128
deep_decoder_params = {
    "output_channels" : 2,                # 1 : grayscale, 2 : complex grayscale, 3 : RGB
    "channels_up"     : [dim_channels]*num_channels,
    "out_sigmoid"     : True,
    "act_funcs"       : ["leakyrelu"]*num_channels,
    "kernel_size"     : 1,                 # when using kernel size one we are not using convolution
    "padding_mode"    : "reflect",
    "upsample_sf"     : 2,                # upsample scale factor
    "upsample_mode"   : "bilinear",
    "upsample_first"  : True,
}

# optimization config for final reconstruction
dd_nepochs             = 10000
dd_optimizer_params    = {"lr": 5e-3, "eps": 1e-8, "weight_decay": 0}
dd_f_optimizer         = lambda net_params, opt_params=dd_optimizer_params : torch.optim.Adam(net_params, **opt_params)
dd_lr_scheduler_params = {"step_size": 100, "gamma": 0.98}
dd_f_lr_scheduler      = lambda optimizer, scheduler_params=dd_lr_scheduler_params : torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

# config adv. loss objective
loss_adv_partial = partial(loss_adv_example,  meas_op = OpA, beta = 1e-3)

# NOTE: same adv. attack method as for DIP, i.e. PAdam_DIP_x
# NOTE: same reconstruction method as for DIP, i.e. _reconstructDIP
# NOTE: same adv. attack loss objective as for DIP, i.e. loss_adv_examle from dip_utils.py
_append_net(
    name = "DeepDecoder no jit",
    info = {
        "name_disp"     : "DeepDecoder w/o noise",
        "name_save"     : "DeepDecoder",
        "plt_color"     : "#000000",
        "plt_marker"    : ".",
        "plt_linestyle" : "dashdot",
        "plt_linewidth" : 2.75,
    },
    net = DeepDecoder(**deep_decoder_params),
    adv_optim  = PAdam_DIP_x,
    rec_config = {
        "niter_adv_optim"         : 1000,
        "reconstruction_method"   : "DeepDecoder",
        "reconstruction_function" : partial(
            _reconstructDIP,
            f_optimizer = dd_f_optimizer,
            f_scheduler = dd_f_lr_scheduler,
            OpA         = OpA,
            epochs      = dd_nepochs,
            sigma_p     = 1/50,
        ),
        "rec_func_adv_noise" : lambda y, xhat : xhat,
        "codomain_distance"  : loss_adv_partial,
        "batch_size"         : 1,
        "net_in_channels"    : deep_decoder_params["channels_up"][0],
        "save_ztilde"        : True,
    }
)

