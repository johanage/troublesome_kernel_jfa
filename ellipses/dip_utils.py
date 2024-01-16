from matplotlib import pyplot as plt
import torch
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import os

def plot_train_DIP(
    img     : torch.tensor,
    img_rec : torch.tensor,
    logging : pd.DataFrame,
    save_fn : str = "DIP_train_metrics.png",
    cmap    : str = "Greys_r",
) -> None:
    num_logging = len(logging.keys())
    fig, axs = plt.subplots(1,3 + num_logging,figsize=(15 + 5*num_logging,5) )
    plot_img = axs[0].imshow(img, cmap=cmap); plot_img_rec = axs[1].imshow(img_rec, cmap=cmap);
    plot_res = axs[2].imshow(torch.sqrt( (img - img_rec)**2), cmap=cmap)
    for ax,plot in zip(axs,[plot_img, plot_img_rec, plot_res]):
        divider = mal(ax)
        cax     = divider.append_axes("right", size="5%", pad = 0.05)
        fig.colorbar(plot, cax=cax)
    axs[0].set_title("Original image")
    axs[1].set_title("Reconstructed image")
    axs[2].set_title("$\ell_2$-reconstruction error")
    # plot logging variables
    for i, key in enumerate(logging.keys() ):
        val = logging[key]
        if key in ["loss", "lr"]:
            # plot logloss for better inspection
            val = torch.log(torch.tensor(val))
            key = "log-" + key
        axs[3 + i].plot(val)
        axs[3 + i].set_title(key)
    
    # savefig
    fig.tight_layout()
    fig.savefig(os.getcwd() + "/" +  save_fn, bbox_inches = "tight")

def get_img_rec(sample, z_tilde, model):
    img = torch.sqrt(sample[0]**2 + sample[1]**2).to("cpu")
    reconstruction = model.forward(z_tilde)
    img_rec = torch.sqrt(reconstruction[0,0]**2 + reconstruction[0,1]**2).detach().to("cpu")
    return img, img_rec, reconstruction

def center_scale_01(image):
    return (image - image.min() )/ (image.max() - image.min() )

import operators
def loss_adv(
    adv_noise : torch.Tensor, 
    xhat      : torch.Tensor,  
    x         : torch.Tensor, 
    meas_op   : operators.Fourier, 
    beta      : float,
) -> torch.Tensor:
    """
    Loss function used in the first step to acquire the adversarial noise.
       l_adv = ||A xhat - (Ax + delta)||_2^2 - beta * || x - xhat||_2^2
    
    Args:
    - adv_noise : adversarial noise
    - xhat      : reconstructed image
    - meas_op   : measurement operator A
    - beta      : parameter of penalizing closeness between orig. image x and reconstructed image xhat
    Out:
    - l_adv as written above
    """
    yhat = meas_op(xhat)
    y_perturbed = meas_op(x) + adv_noise
    recerr = xhat - x
    return (yhat - y_perturbed).pow(2).sum() - beta * recerr.pow(2).sum()

from types import GeneratorType
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
