# Robustness test configuration for
# - Supervised learning
# - Deep image prior - serving as untrained network/ unsupervised method

import os
from typing import Tuple
import matplotlib as mpl
import pandas as pd
import torch
import torchvision
from piq import psnr, ssim

from data_management import IPDataset, SimulateMeasurements, ToComplex
from networks import UNet
from operators import (
    Fourier,
    Fourier_matrix as Fourier_m,
    LearnableInverterFourier,
    RadialMaskFunc,
    MaskFromFile,
    noise_gaussian,
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
methods = pd.DataFrame(columns=["name", "info", "reconstr", "attacker", "net"])
methods = methods.set_index("name")
# reference noise function is set to gaussian additive noise, see operators.py for description
noise_ref = noise_gaussian

# ----- set up net attacks --------

# the actual reconstruction method for any net
def _reconstructNet(y, noise_rel, net):
    return net.forward(y)

# attack function for any net
def _attackerNet(
    x0         : torch.Tensor,
    noise_rel  : float,
    net        : torch.nn.Module,
    yadv_init  : torch.Tensor = None,
    batch_size : int = 3
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
    - x0        : ground truth image
    - noise_rel : noise level relative to the ground truth image
    - net       : DNN to use
    - yadv_init : initial adversarial measurements
    
    Out:
    - yadv : adversarial measurement
    - yref : refrence measuruement (measurement with gaussian noise)
    - y0   : noiseless measurment
    
    """
    # compute noiseless measurements
    y0 = OpA(x0)

    if noise_rel == 0.0:
        return y0, y0, y0
    # compute absolute noise levels
    noise_level = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    # compute noisy measurements for reference
    yref = noise_ref(OpA(x0), noise_level)  # noisy measurements
    # attack parameters
    adv_init_fac = 3.0 * noise_level
    adv_param = {
        "codomain_dist" : _complexloss,
        "domain_dist"   : None,
        "mixed_dist"    : None,
        "weights"       : (1.0, 1.0, 1.0),
        "optimizer"     : PAdam,
        "projs"         : None,
        "iter"          : 1000,
        "stepsize"      : 5e0,
    }
    # compute initialization
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)

    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()

    for idx_batch in range(0, yadv.shape[0], batch_size):
        print(
            "Attack for samples "
            + str(list(range(idx_batch, idx_batch + batch_size)))
        )

        adv_param["projs"] = [
            lambda y: proj_l2_ball(
                y,
                y0[idx_batch : idx_batch + batch_size, ...],
                noise_level[idx_batch : idx_batch + batch_size, ...],
            )
        ]
        # perform attack
        yadv[idx_batch : idx_batch + batch_size, ...] = untargeted_attack(
            lambda y: _reconstructNet(y, 0.0, net),
            yadv[idx_batch : idx_batch + batch_size, ...]
            .clone()
            .requires_grad_(True),
            y0[idx_batch : idx_batch + batch_size, ...],
            t_out_ref=x0[idx_batch : idx_batch + batch_size, ...],
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
    # net_loaded.freeze()
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.eval()
    return network

# methods dataframe append function for each net configuration
def _append_net(
    name : str, 
    info : dict, 
    net  : torch.nn.Module
) -> None:
    """
    Appends reconstruction method to methods dataframe defined locally above.
    
    Args:
    - name : name of the method
    - info : metadata of the method for logging purposes
    - net  : network loaded with pretrained parameters

    Out: Nothing
    """
    methods.loc[name] = {
        "info": info,
        "reconstr": lambda y, noise_rel: _reconstructNet(y, noise_rel, net),
        "attacker": lambda x0, noise_rel, yadv_init=None: _attackerNet(
            x0, noise_rel, net, yadv_init=yadv_init
        ),
        "net": net,
    }

# ----- DIP UNet configuration -----
dip_unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : None, 
}

_append_net(
    "DIP UNet jit",
    {
        "name_disp": "DIP UNet w/ low noise",
        "name_save": "dip_unet_jit",
        "plt_color": "#023eff",
        "plt_marker": "o",
        "plt_linestyle": "--",
        "plt_linewidth": 2.75,
    },
    _load_net(
        f"{config.RESULTS_PATH}/DIP/"
        + "DIP_UNet_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1_last.pt",
        UNet,
        dip_unet_params,
    ),
)

# ----- DIP UNet configuration -----
supervised_unet_params = {
    "in_channels"   : 2,
    "drop_factor"   : 0.0,
    "base_features" : 32,
    "out_channels"  : 2,
    "operator"      : OpA_m,
    "inverter"      : inverter,
}
_append_net(
    "Supervised UNet no jit",
    {
        "name_disp": "Supervised UNet w/o noise",
        "name_save": "unet_no_jit",
        "plt_color": "#023eff",
        "plt_marker": "x",
        "plt_linestyle": "--",
        "plt_linewidth": 2.75,
    },
    _load_net(
        f"{config.RESULTS_PATH}/supervised/circ_sr0.25/Fourier_UNet_no_jitter_brain_fastmri_256train_phase_2/"
        + "model_weights.pt",
        UNet,
        supervised_unet_params,
    ),
)
