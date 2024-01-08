import os
from typing import Tuple, Union, List, Optional

import numpy as np
import pandas as pd
import torch

import config

from find_adversarial import PAdam, untargeted_attack
from networks import IterativeNet, Tiramisu, UNet
from operators import (
    Fourier,
    Fourier_matrix,
    LearnableInverterFourier,
    RadialMaskFunc,
    TVAnalysisPeriodic,
    noise_gaussian,
    proj_l2_ball,
)
from reconstruction_methods import admm_l1_rec_diag


# ------ setup ----------
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- operators -----
mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
OpA = Fourier(mask)
OpTV = TVAnalysisPeriodic(config.n, device=device)

# ----- methods --------
methods = pd.DataFrame(columns=["name", "info", "reconstr", "attacker", "net"])
methods = methods.set_index("name")

noise_ref = noise_gaussian

def _get_gs_param(noise_rel):
    lam =  0.001;
    rho = 0.5;
    return lam, rho 


# the actual reconstruction method
def _reconstructL1(y, noise_rel):
    if torch.is_tensor(noise_rel):
        lam, rho = _get_gs_param(noise_rel.numpy())
    else:
        lam, rho = _get_gs_param(noise_rel)
    x, _ = admm_l1_rec_diag(
        y,
        OpA,
        OpTV,
        OpA.adj(y),
        OpTV(OpA.adj(y)),
        lam,
        rho,
        iter=5000,
        silent=True,
    )
    return x


# the reconstruction method used for the L1 attack
# (less iterations due to high computational costs)
def _reconstructL1_adv(y, lam, rho, x0, z0):
    x, _ = admm_l1_rec_diag(
        y, OpA, OpTV, x0, z0, lam, rho, iter=200, silent=True
    )
    return x


# loss
mseloss = torch.nn.MSELoss(reduction="sum")


def _complexloss(reference, prediction):
    loss = mseloss(reference, prediction)
    return loss


# attack function for L1
def _attackerL1(x0, noise_rel, yadv_init=None, batch_size=6):

    # compute noiseless measurements
    y0 = OpA(x0)

    if noise_rel == 0.0:
        return y0, y0, y0

    # compute absolute noise levels
    noise_level = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    # compute noisy measurements for reference
    yref = noise_ref(OpA(x0), noise_level)
    # attack parameters
    adv_init_fac = 3.0 * noise_level
    adv_param = {
        "codomain_dist": _complexloss,
        "domain_dist": None,
        "mixed_dist": None,
        "weights": (1.0, 1.0, 1.0),
        "optimizer": PAdam,
        "projs": None,
        "iter": 250,
        "stepsize": 5e0,
    }
    # get ADMM tuning parameters for noise_rel
    lam, rho = _get_gs_param(noise_rel.numpy())

    # compute good start values for _reconstructL1_adv
    x0_adv, z0_adv = admm_l1_rec_diag(
        y0,
        OpA,
        OpTV,
        OpA.adj(y0),
        OpTV(OpA.adj(y0)),
        lam,
        rho,
        iter=5000,
        silent=True,
    )
    print('z0_adv.shape: ', z0_adv.shape)
    
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
            lambda y: _reconstructL1_adv(
                y,
                lam,
                rho,
                x0_adv[idx_batch : idx_batch + batch_size, ...],
                z0_adv[idx_batch : idx_batch + batch_size, ...],
            ),
            yadv[idx_batch : idx_batch + batch_size, ...]
            .clone()
            .requires_grad_(True),
            y0[idx_batch : idx_batch + batch_size, ...],
            t_out_ref=x0[idx_batch : idx_batch + batch_size, ...],
            **adv_param
        ).detach()

    return yadv, yref, y0


methods.loc["L1"] = {
    "info": {
        "name_disp": "TV$[\\lambda=0.001]$",
        "name_save": "tv",
        "plt_color": "#e8000b",
        "plt_marker": "o",
        "plt_linestyle": "-",
        "plt_linewidth": 2.75,
    },
    "reconstr": _reconstructL1,
    "attacker": lambda x0, noise_rel, yadv_init=None: _attackerL1(
        x0, noise_rel, yadv_init=yadv_init
    ),
    "net": None,
}
methods.loc["L1", "net"] = None


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
    Attacker function for any DNN
    -----------------------------------------------------------
    x0        : ground truth image
    noise_rel : noise level relative to the ground truth image
    net       : DNN to use
    yadv_init : initial adversarial measurements
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
        "codomain_dist": _complexloss,
        "domain_dist": None,
        "mixed_dist": None,
        "weights": (1.0, 1.0, 1.0),
        "optimizer": PAdam,
        "projs": None,
        "iter": 1000,
        "stepsize": 5e0,
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


# ----- load nets -----

# create a net and load weights from file
def _load_net(path, subnet, subnet_params, it_net_params):
    subnet = subnet(**subnet_params).to(device)
    it_net = IterativeNet(subnet, **it_net_params).to(device)
    it_net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    it_net.freeze()
    it_net.eval()
    return it_net

# methods dataframe append function for each net configuration
def _append_net(name, info, net):
    methods.loc[name] = {
        "info": info,
        "reconstr": lambda y, noise_rel: _reconstructNet(y, noise_rel, net),
        "attacker": lambda x0, noise_rel, yadv_init=None: _attackerNet(
            x0, noise_rel, net, yadv_init=yadv_init
        ),
        "net": net,
    }
    pass


# ----- UNets -----
unet_params = {
    "in_channels": 2,
    "drop_factor": 0.0,
    "base_features": 32,
    "out_channels": 2,
}

_append_net(
    "UNet it jit",
    {
        "name_disp": "ItNet w/ high noise",
        "name_save": "unet_it_jit",
        "plt_color": "#023eff",
        "plt_marker": "o",
        "plt_linestyle": "--",
        "plt_linewidth": 2.75,
    },
    _load_net(
        f"{config.RESULTS_PATH}/Fourier_UNet_it_jitter_v4_train_phase_2/"
        + "model_weights.pt",
        UNet,
        unet_params,
        {
            "num_iter": 8,
            "lam": 8 * [0.1],
            "lam_learnable": False,
            "final_dc": True,
            "resnet_factor": 1.0,
            "operator": Fourier_matrix(mask),
            "inverter": LearnableInverterFourier(config.n, mask, learnable=False),
        },
    ),
)

_append_net(
    "UNet it jit mod",
    {
        "name_disp": "ItNet w/ low noise",
        "name_save": "unet_it_jit_mod",
        "plt_color": "turquoise",
        "plt_marker": "o",
        "plt_linestyle": "-",
        "plt_linewidth": 2.75,
    },
    _load_net(
        f"{config.RESULTS_PATH}/Fourier_UNet_it_jit-nojit_train_phase_1/"
        + "model_weights.pt",
        UNet,
        unet_params,
        {
            "num_iter": 8,
            "lam": 8 * [0.1],
            "lam_learnable": False,
            "final_dc": True,
            "resnet_factor": 1.0,
            "operator": Fourier_matrix(mask),
            "inverter": LearnableInverterFourier(config.n, mask, learnable=False),
        },
    ),
)

_append_net(
    "UNet it no jit",
    {
        "name_disp": "ItNet w/o noise",
        "name_save": "unet_it_no_jit",
        "plt_color": "#1ac938",
        "plt_marker": "o",
        "plt_linestyle": "-.",
        "plt_linewidth": 2.75,
    },
    _load_net(
        f"{config.RESULTS_PATH}/Fourier_UNet_it_no_jitter_train_phase_2/"
        + "model_weights.pt",
        UNet,
        unet_params,
        {
            "num_iter": 8,
            "lam": 8 * [0.1],
            "lam_learnable": False,
            "final_dc": True,
            "resnet_factor": 1.0,
            "operator": Fourier_matrix(mask),
            "inverter": LearnableInverterFourier(config.n, mask, learnable=False),
        },
    ),
)

