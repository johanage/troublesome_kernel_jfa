# Robustness test configuration for
# - Supervised learning
# - Deep image prior - serving as untrained network/ unsupervised method

import os
from typing import Tuple, Type, Callable
import matplotlib as mpl
import pandas as pd, torch, torchvision, numpy as np
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
    proj_l2_ball,
)
from find_adversarial import (
    PGD,
    PAdam,
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
methods = pd.DataFrame(columns=["name", "info", "reconstr", "attacker", "net"])
methods = methods.set_index("name")
# reference noise function is set to gaussian additive noise, see operators.py for description
noise_ref = noise_gaussian

# ----- set up net attacks --------

# loss
mseloss = torch.nn.MSELoss(reduction="sum")
def _complexloss(reference, prediction):
    loss = mseloss(reference, prediction)
    return loss

# the actual reconstruction method for any net
#def _reconstructNet(y, noise_rel, net):
def _reconstructNet(y, net):
    return net.forward(y)

from types import GeneratorType
mseloss = torch.nn.MSELoss(reduction="sum")
def loss_func(pred, tar):
    return (
        mseloss(pred, tar) / pred.shape[0]
    )

from tqdm import tqdm
def _reconstructDIP(
    y0          : torch.Tensor,
    net         : Type[torch.nn.Module], 
    f_optimizer : Callable[GeneratorType, Type[torch.optim.Optimizer]],
    f_scheduler : Callable[Type[torch.optim.Optimizer], Type[torch.optim.lr_scheduler.LRScheduler]],
    z_tilde     : torch.Tensor,
    epochs      : int,
    loss_func   : Callable = loss_func,
    sigma_p     : float    = 1/30,
) -> torch.Tensor:
    """
    DIP reconstruction algorithm.
    Args:
    - y0          : The noiseless measurements
    - net         : DIP network
    - f_optimizer : Function with predetermined optimization params
                    that takes net parameters as input. 
    - f_scheudler : Function with predetermined lr scheduler params
                    that takes optimizer object as input.
    - z_tilde     : The fixed random input vector of the DIP net, 
                    this is decoded to an image.
    - sigma_p     : DIP uses jittering, magnitude of Gaussian jittering noise.
    """
    # set device for net and z_tilde
    net.to(device); z_tilde = z_tilde.to(device)
    # activate train-based layers
    net.train()
    # set parameters to be trainable
    for param in net.parameters():
        param.requires_grad = True
    optimizer = f_optimizer(net.parameters())
    scheduler = f_scheduler(optimizer)
    # progressbar setup
    progress_bar = tqdm(
        desc="Train DIP ",
        total=epochs,
    )

    for epoch in range(epochs):
        # set gradients to zero
        optimizer.zero_grad()
        # add gaussian noise to noise input according to Ulyanov et al 2020
        additive_noise = sigma_p*torch.randn(z_tilde.shape).to(device)
        model_input = z_tilde + additive_noise
        model_input.to(device)
        # pred_img = Psi_theta(z_tilde + additive_noise)
        pred_img = net.forward(model_input)
        # pred = A G(z_tilde, theta)
        pred = OpA(pred_img)
        loss = loss_func(pred, y0)
        loss.backward()
        # update weights
        optimizer.step()
        # update lr
        scheduler.step()
        # udate progress bar
        progress_bar.update(1)
        progress_bar.set_postfix(
            **net._add_to_progress_bar({"loss": loss.item()})
        )

    # ----------- Image reconstruction -------------------
    # deactivate train-based layers
    net.eval()
    # freeze parameters
    for param in net.parameters():
        param.requires_grad = False
    # no jittering used in last reconstruction - only during training
    pred_img = net.forward(z_tilde)
    return pred_img

# attack function for any net
def _attackerNet(
    x0         : torch.Tensor,
    noise_rel  : float,
    net        : torch.nn.Module,
    net_dinput : int = 2,
    yadv_init  : torch.Tensor = None,
    rec        : Callable = None,
    batch_size : int = 3,
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
    - rec       : reconstruction function
    - batch_size: batch size
    Out:
    - yadv : adversarial measurement
    - yref : refrence measuruement (measurement with gaussian noise)
    - y0   : noiseless measurment
    """
    # set reconstruction function
    if rec is None:
        rec = lambda y: _reconstructNet(y, net)
    # this can be changed to string instead of boolean condition if more learning paradigms are added
    else:
        rec = partial(rec, net=net)
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
    # set init adversarial meas example if previous lower noise level was computed
    if yadv_init is not None:
        yadv[0 : yadv_init.shape[0], ...] = yadv_init.clone().detach()

    for idx_batch in range(0, yadv.shape[0], batch_size):
        print(
            "Attack for samples "
            + str(list(range(idx_batch, idx_batch + batch_size)))
        )
        y0_batch = y0[idx_batch : idx_batch + batch_size, ...]
        if rec is not None:
            # z_tilde ~ U([0,noise_mag]) with shape of (batch_size, input_channel, image_w, image_h)
            noise_mag = 0.1
            z_tilde   = noise_mag * torch.rand((batch_size, net_dinput) + x0.shape[-2:])
            z_tilde.to(device)
            # update reconstruction including the z_tilde vector for each measurement
            rec = partial(rec, z_tilde = z_tilde)
        # set l2-ball projection
        # - centered at y0[idx_batch : idx_batch + batch_size, ...]
        # - with radius given by noise_level[idx_batch : idx_batch + batch_size, ...]
        adv_param["projs"] = [
            lambda y: proj_l2_ball(
                y,
                #y0[idx_batch : idx_batch + batch_size, ...],
                y0_batch,
                noise_level[idx_batch : idx_batch + batch_size, ...],
            )
        ]
        # perform untargeted attack
        yadv[idx_batch : idx_batch + batch_size, ...] = untargeted_attack(
            # default | rec = lambda y: _reconstructNet(y, net),
            func      = rec,
            t_in_adv  = yadv[idx_batch : idx_batch + batch_size, ...].clone().requires_grad_(True),
            #t_in_ref  = y0[idx_batch : idx_batch + batch_size, ...],
            t_in_ref  = y0_batch,
            t_out_ref = x0[idx_batch : idx_batch + batch_size, ...],
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
    net  : torch.nn.Module,
    rec  : Callable = None,
) -> None:
    """
    Appends reconstruction method to methods dataframe defined locally above.
    
    Args:
    - name : name of the method
    - info : metadata of the method for logging purposes
    - net  : network loaded with pretrained parameters
    - rec  : explicit reconstruction method
    Out: Nothing
    """
    if rec is None:
        # lambda y, noise_rel: _reconstructNet(y, noise_rel, net),
        rec = partial(_reconstructNet, net=net)
    methods.loc[name] = {
        "info"     : info,
        "reconstr" : rec,
        "attacker" : lambda x0, noise_rel, yadv_init=None, rec=None: _attackerNet(
            x0, noise_rel, net, yadv_init = yadv_init, rec = rec,
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

dip_optimizer_params    = {"lr": 1e-4, "eps": 1e-8, "weight_decay": 0}
dip_f_optimizer           = lambda net_params, opt_params=dip_optimizer_params : torch.optim.Adam(net_params, **opt_params)
dip_lr_scheduler_params = {"step_size": 100, "gamma": 0.96} 
dip_f_lr_scheduler        = lambda optimizer, scheduler_params=dip_lr_scheduler_params : torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

dir_val = "/mn/nam-shub-02/scratch/vegarant/pytorch_datasets/fastMRI/val/"

# same as DIP
from operators import to_complex
z_tilde = torch.load(os.getcwd() + "/adv_attack_dip/z_tilde.pt")

from functools import partial
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
    rec = partial(
        _reconstructDIP, 
        f_optimizer = dip_f_optimizer,
        f_scheduler = dip_f_lr_scheduler,
        epochs      = 1000,
    ),
)

# ----- Supervised UNet configuration -----
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
