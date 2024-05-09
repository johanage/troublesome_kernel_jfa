import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from piq import psnr, ssim
from data_management import IPDataset
from find_adversarial import err_measure_l2, grid_attack
from operators import to_magnitude, to_complex

# ----- load configuration -----
import config 
import config_robustness_fourier_SL_DIP as cfg_rob  
from config_robustness_fourier_SL_DIP import methods 

# ------ general setup ----------
device = cfg_rob.device
torch.manual_seed(1)

save_plot = True
# select samples
#sample = "00042" # fastMRI val sample with CANCER-text
sample = "21" # ellipses val sample with text
save_path = os.path.join(config.RESULTS_PATH, "attacks", "example_S%s"%sample)

# dynamic range for plotting & similarity indices
v_min = 0.0
v_max = 0.9

err_measure = err_measure_l2

# select reconstruction methods
methods_include = [
    "DeepDecoder no jit",
    "DIP UNet no jit",
    #"DIP UNet no jit 2/3 iterations",
    #"DIP UNet no jit 1/3 iterations",
    #"DIP UNet jit",
    #"Supervised UNet no jit",
    #"Supervised UNet jit",
    #"Supervised UNet jit low noise",
    #"Supervised UNet jit mod",
    #"Supervised UNet jit very high noise",
    #'L1',
]


methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = [
    #"DeepDecoder no jit",
    #"DIP UNet no jit",
    #"DIP UNet no jit 2/3 iterations",
    #"DIP UNet no jit 1/3 iterations",
    #"DIP UNet jit",
    #"Supervised UNet no jit",
    #"Supervised UNet jit",
    #"Supervised UNet jit low noise",
    #"Supervised UNet jit mod",
    #"Supervised UNet jit very high noise",
    #'L1',
]


# select sample
#single_im = torch.load(os.path.join(config.DATA_PATH, "val",  f'sample_{sample}_text.pt')) # fastMRI
single_im = torch.load(os.path.join(config.TOY_DATA_PATH, "val",  f'sample_{sample}_text.pt')) # ellipses dataset
single_im1 = single_im.unsqueeze(0);

X_0 = to_complex(single_im1.to(device)).unsqueeze(0)

it_init = 1
print((X_0.ndim-1)*(1,))
X_0 = X_0.repeat(it_init, *((X_0.ndim - 1) * (1,)))
print('X_0.shape: ', X_0.shape)
Y_0 = cfg_rob.OpA(X_0)

X_0_cpu = X_0.cpu()



# ----- plotting -----
def _implot(sub, im, vmin=v_min, vmax=v_max, **imshow_kwargs):
    if im.shape[-3] == 2:  # complex image
        image = sub.imshow(
            torch.sqrt(im.pow(2).sum(-3))[0,:,:].detach().cpu(),
            vmin=vmin,
            vmax=vmax,
            **imshow_kwargs
        )
    else:  # real image
        image = sub.imshow(im[0, 0, :, :].detach().cpu(), vmin=vmin, vmax=vmax, **imshow_kwargs)

    image.set_cmap("gray")#("gray")
    sub.set_xticks([])
    sub.set_yticks([])
    return image

# LaTeX typesetting
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

# perform reconstruction
for (idx, method) in methods.iterrows():
    print("plotting adversarial text reconstructions for ", method["info"]["name_disp"])
    if idx not in methods_no_calc:
        # for DIP networks we need to learn pre-trained network
        if "DIP" in method.name:
            z_tilde = 0.1 * torch.rand( (1, method.rec_config["net_in_channels"],) + X_0.shape[-2:] )
            torch.save(z_tilde, os.path.join(config.RESULTS_PATH, "attacks", "DIP_x_UNet_ztilde_adv_text.pt") )
            # save initialised model weights to be loaded in each reconstruction at each noise level
            torch.save(method["net"].state_dict(), os.path.join(config.RESULTS_PATH, "attacks", "DIP_x_UNet_init_weights_adv_text.pt") )
            # note that Y0 is Y0.shape[0] identical samples
            X_rec = method.reconstr(y0 = Y_0[:1], net = method["net"], z_tilde = z_tilde)
            #method.rec_config["xhat0"] = X_rec.repeat( Y_0.shape[0], *((X_0.ndim -1) * (1,)) )

        # for DeepDecoder networks we need to learn pre-trained network
        if "DeepDecoder" in method.name:
            # use same input name convention as for DIP to save space
            z_tilde = 10 * torch.rand( (1, method.rec_config["net_in_channels"],) + tuple([d//2**(method["net"].nscales-1) for d in X_0.shape[-2:]]) )
            torch.save(z_tilde, os.path.join(config.RESULTS_PATH, "attacks", "DeepDecoder_ztilde_adv_text.pt") )
            # save initialised model weights to be loaded in each reconstruction at each noise level
            torch.save(method["net"].state_dict(), os.path.join(config.RESULTS_PATH, "DeepDecoder_init_weights_adv_text.pt") )
            # note that Y0 is Y0.shape[0] identical samples
            X_rec = method.reconstr(y0 = Y_0[:1], net = method["net"], z_tilde = z_tilde)
            #method.rec_config["xhat0"] = Xhat.repeat( Y_0.shape[0], *((X_0.ndim -1) * (1,)) )
        
        if "Supervised" in method.name:
            X_rec = method.reconstr(Y_0)#, 0)
        
        # save image using torchvision + PIL
        save_image(X_rec.norm(p=2, dim=(0,1)), os.path.join(save_path,
            "image_example_S{}_adv_err_".format(sample)
            + method["info"]["name_save"]
            + "_text.png"
        ))
        # compute the l2-reconstruction error
        rec_err = err_measure(X_rec, X_0);
        print(f'{idx}: rel l2 error: {rec_err}');

        fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
        extent = (0, 255, 0, 255)
        im = _implot(ax, X_rec, **{"extent" : extent, "origin" : "upper"})
        """
        ax.text(
            242,
            6,
            "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                rec_err * 100
            ),
            fontsize=10,
            color="white",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        # make inset axis
        axins = ax.inset_axes(
            [0.05, 0.75, 0.4,0.2],
        )
        axins.set_xlim(143,180)
        axins.set_ylim(135,150)
        # plot the reconstruction in the inset 
        _implot(axins, X_rec, **{"extent" : extent, "origin" : "upper"})
        axins.spines["bottom"].set_color("#a1c9f4")
        axins.spines["top"].set_color("#a1c9f4")
        axins.spines["left"].set_color("#a1c9f4")
        axins.spines["right"].set_color("#a1c9f4")
        #axins.invert_yaxis()
        #zoom_rect = mark_inset(ax, axins, loc1=1, loc2=3, edgecolor="#a1c9f4")
        ax.indicate_inset_zoom(axins, edgecolor="#a1c9f4")
        """
        if save_plot:
            fig.savefig(
                os.path.join(
                    save_path,
                    "fig_example_S{}_adv_".format(sample)
                    + method["info"]["name_save"]
                    + "_text.pdf"
                ),
                bbox_inches="tight",
                pad_inches=0,
            )

        # not saved
        fig.suptitle(
            method["info"]["name_disp"] + " for unseen detail"
        )

        # error plot
        fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
        im = _implot(
            ax,
            (to_magnitude(X_rec) - to_magnitude(X_0)).abs(),
            vmin=0.0,
            vmax=0.4,
        )

        if save_plot:
            fig.savefig(
                os.path.join(
                    save_path,
                    "fig_example_S{}_adv_err_".format(sample)
                    + method["info"]["name_save"]
                    + "_text.pdf"
                ),
                bbox_inches="tight",
                pad_inches=0,
            )



