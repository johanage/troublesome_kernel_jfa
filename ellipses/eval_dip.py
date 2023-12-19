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
