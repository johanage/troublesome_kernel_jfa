import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import rc
from piq import psnr, ssim

from data_management import IPDataset
from find_adversarial import err_measure_l2, grid_attack
from operators import to_complex


# ----- load configuration -----
import config  
import config_robustness_fourier_SL_DIP as cfg_rob  
methods = cfg_rob.methods
# select reconstruction methods
methods_include = [
    #"DIP UNet no jit",
    #"DIP UNet no jit 2/3 iterations",
    #"DIP UNet no jit 1/3 iterations",
    #"DIP UNet jit",
    "Supervised UNet no jit",
    "Supervised UNet jit",
    "Supervised UNet jit low noise",
    "Supervised UNet jit mod",
    "Supervised UNet jit very high noise",
    #'L1',
    #"UNet it no jit",
    #"UNet it jit mod",
    #"UNet it jit",
]
methods_plot = methods_include #["L1", "UNet it jit mod",  "UNet it no jit"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = [
    "DIP UNet no jit",
    "DIP UNet no jit 2/3 iterations",
    "DIP UNet no jit 1/3 iterations",
    "DIP UNet jit",
    "Supervised UNet no jit",
    "Supervised UNet jit",
    #"Supervised UNet jit low noise",
    "Supervised UNet jit mod",
    "Supervised UNet jit very high noise",
    #'L1',
    #"UNet it jit",
    #"UNet it no jit",
    #"UNet it jit mod",
]
# ------ general setup ----------
device = cfg_rob.device
save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_results = os.path.join(save_path, "dip_supervised_table_adv.pkl")
# plot config
do_plot = True
save_plot = True
save_table = True

# ----- attack setup -----
# select samples
# selecting sample numbers 50, ..., 59
samples = tuple(range(50, 60))

it_init = 6
keep_init = 3

# select range relative noise
noise_rel = torch.tensor([0.00, 0.02, 0.04, 0.06, 0.08])

# select measure for reconstruction error
err_measure = err_measure_l2

# ----- perform attack -----
# select samples
val_data = IPDataset("val", config.DATA_PATH)
X_0 = torch.stack([val_data[s][0] for s in samples])
X_0 = to_complex(X_0.to(device))
Y_0 = cfg_rob.OpA(X_0)


# create result table
results = pd.DataFrame(
    columns=[
        "name",
        "X_adv_err",
        "X_ref_err",
        "X_adv_psnr",
        "X_ref_psnr",
        "X_adv_ssim",
        "X_ref_ssim",
    ]
)
results.name = methods.index
results = results.set_index("name")
# load existing results from file
if os.path.isfile(save_results):
    results_save = pd.read_pickle(save_results)
    for idx in results_save.index:
        if idx in results.index:
            results.loc[idx] = results_save.loc[idx]
else:
    results_save = results

s_len = X_0.shape[0]

for s in range(s_len):
    X_0_s = X_0[s : s + 1, ...].repeat(
        it_init, *((X_0.ndim - 1) * (1,))
    )

# perform attacks
for (idx, method) in methods.iterrows():
    if idx not in methods_no_calc:

        s_len = X_0.shape[0]
        results.loc[idx].X_adv_err = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ref_err = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_adv_psnr = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ref_psnr = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_adv_ssim = torch.zeros(len(noise_rel), s_len)
        results.loc[idx].X_ref_ssim = torch.zeros(len(noise_rel), s_len)

        for s in range(s_len):
            print("Sample: {}/{}".format(s + 1, s_len))
            # sample at index s repeated it_init times 
            X_0_s = X_0[s : s + 1, ...].repeat(
                it_init, *((X_0.ndim - 1) * (1,))
            )
            Y_0_s = Y_0[s : s + 1, ...].repeat(
                it_init, *((Y_0.ndim - 1) * (1,))
            )
            # grid attack using noise_rel noise levels 
            (
                X_adv_err_cur,
                X_ref_err_cur,
                X_adv_cur,
                X_ref_cur,
                _,
                _,
            ) = grid_attack(
                method,
                noise_rel,
                X_0_s,
                Y_0_s,
                store_data  = True,
                keep_init   = keep_init,
                err_measure = err_measure,
            )
            # store max current adversarial error
            (
                results.loc[idx].X_adv_err[:, s],
                idx_max_adv_err,
            ) = X_adv_err_cur.max(dim=1)
            # store current mean of reference error
            results.loc[idx].X_ref_err[:, s] = X_ref_err_cur.mean(dim=1)


            for idx_noise in range(len(noise_rel)):
                idx_max = 0;
                a11 = X_adv_cur[idx_noise, idx_max, ...];
                
                idx_max = idx_max_adv_err[idx_noise]
                results.loc[idx].X_adv_psnr[idx_noise, s] = psnr(
                    torch.clamp(
                        X_adv_cur[idx_noise, idx_max, ...].unsqueeze(0), 
                        min=0,
                        max=1
                    ),
                    X_0_s[0, ...].unsqueeze(0).cpu(),
                    data_range=1.0,
                    reduction="none",
                )
                results.loc[idx].X_ref_psnr[idx_noise, s] = psnr(
                    torch.clamp(
                        X_ref_cur[idx_noise, idx_max, ...].unsqueeze(0), 
                        min=0, 
                        max=1
                    ),
                    X_0_s[0, ...].unsqueeze(0).cpu(),
                    data_range=1.0,
                    reduction="mean",
                )
                results.loc[idx].X_adv_ssim[idx_noise, s] = ssim(
                    torch.clamp(
                        X_adv_cur[idx_noise, idx_max, ...].unsqueeze(0), 
                        min=0,
                        max=1
                    ),
                    X_0_s[0, ...].unsqueeze(0).cpu(),
                    data_range=1.0,
                )
                results.loc[idx].X_ref_ssim[idx_noise, s] = ssim(
                    torch.clamp(
                        X_ref_cur[idx_noise, idx_max, ...].unsqueeze(0), 
                        min=0, 
                        max=1
                    ),
                    X_0_s[0, ...].unsqueeze(0).cpu(),
                    data_range=1.0,
                )

# save results
for idx in results.index:
    results_save.loc[idx] = results.loc[idx]
os.makedirs(save_path, exist_ok=True)
results_save.to_pickle(save_results)

# ----- plotting -----

if do_plot:
    # LaTeX typesetting
    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("text", usetex=True)

    # +++ visualization of table +++
    fig, ax         = plt.subplots(clear=True, figsize=(6, 3), dpi=200)
    fig_std, ax_std = plt.subplots(clear=True, figsize=(6, 3), dpi=200)

    for (idx, method) in methods.loc[methods_plot].iterrows():
        if idx in methods_plot:
            print(f'idx: {idx}, method: {method}, res: {results.loc[idx].X_adv_err}')
            err_mean = results.loc[idx].X_adv_err[:, :].mean(dim=-1)
            err_std  = results.loc[idx].X_adv_err[:, :].std(dim=-1)
            kwargs_plot = {
                "linestyle" : method["info"]["plt_linestyle"],
                "linewidth" : method["info"]["plt_linewidth"],
                "marker"    : method["info"]["plt_marker"],
                "color"     : method["info"]["plt_color"],
                "label"     : method["info"]["name_disp"],
            }
            ax.plot(
                noise_rel,
                err_mean,
                **kwargs_plot,
            )
            ax_std.plot(noise_rel, err_std, **kwargs_plot)
            fill_between_methods = (None,)#("L1", "UNet it jit mod", "UNet it jit", "UNet it no jit", "DIP UNet no jit", "Supervised UNet no jit", "Supervised UNet jit", "Supervised UNet jit low noise")
            if idx in fill_between_methods:
                ax.fill_between(
                    noise_rel,
                    err_mean + err_std,
                    err_mean - err_std,
                    alpha = 0.10,
                    color = method["info"]["plt_color"],
                )

    ax.set_yticks(np.arange(0, 1, step=0.05))
    ax.set_ylim((0.05, 0.18))
    for a in (ax, ax_std):
        a.set_xticklabels(["{:,.0%}".format(x) for x in ax.get_xticks()])
        a.set_yticklabels(["{:,.0%}".format(x) for x in ax.get_yticks()])
        a.legend(loc="upper left", fontsize=12)

    if save_plot:
        fig.savefig(
            #os.path.join(save_path, "fig_table_adv.pdf"), bbox_inches="tight"
            #os.path.join(save_path, "fig_table_adv_dip.pdf"), bbox_inches="tight"
            os.path.join(save_path, "fig_table_adv_supervised.pdf"), bbox_inches="tight"
        )
        fig_std.savefig(
            #os.path.join(save_path, "fig_table_adv_std.pdf"), bbox_inches="tight"
            #os.path.join(save_path, "fig_table_adv_std_dip.pdf"), bbox_inches="tight"
            os.path.join(save_path, "fig_table_adv_std_supervised.pdf"), bbox_inches="tight"
        )
    plt.show()

if save_table:
    df = results.applymap(
        lambda res: {"mean": res.mean(dim=-1), "std": res.std(dim=-1)}
    )

    # split adv and ref results
    df_adv = df[["X_adv_err", "X_adv_psnr", "X_adv_ssim"]]

    # extract mean and std
    df_adv_mean = (
        df_adv.stack()
        .apply(pd.Series)["mean"]
        .apply(
            lambda res: pd.Series(
                res,
                index=[
                    "{{{:.1f}\\%}}".format(noise * 100) for noise in noise_rel
                ],
            )
        )
    )
    df_adv_std = (
        df_adv.stack()
        .apply(pd.Series)["std"]
        .apply(
            lambda res: pd.Series(
                res,
                index=[
                    "{{{:.1f}\\%}}".format(noise * 100) for noise in noise_rel
                ],
            )
        )
    )

    # find best method per noise level and metric
    best_adv_l2 = df_adv_mean.xs("X_adv_err", level=1).idxmin()
    best_adv_ssim = df_adv_mean.xs("X_adv_ssim", level=1).idxmax()
    best_adv_psnr = df_adv_mean.xs("X_adv_psnr", level=1).idxmax()

    # combine mean and std data into "mean\pmstd" strings
    for (idx, method) in methods.iterrows():
        df_adv_mean.loc[idx, "X_adv_err"] = df_adv_mean.loc[
            idx, "X_adv_err"
        ].apply(lambda res: res * 100)
        df_adv_std.loc[idx, "X_adv_err"] = df_adv_std.loc[
            idx, "X_adv_err"
        ].apply(lambda res: res * 100)
    df_adv_combined = df_adv_mean.combine(
        df_adv_std,
        lambda col1, col2: col1.combine(
            col2, lambda el1, el2: "${:.2f} \\pm {:.2f}$".format(el1, el2)
        ),
    )

    # format best value per noise level and metric as bold
    for col, idx in best_adv_l2.iteritems():
        df_adv_combined.at[(idx, "X_adv_err"), col] = (
            "\\bfseries " + df_adv_combined.at[(idx, "X_adv_err"), col]
        )
    for col, idx in best_adv_ssim.iteritems():
        df_adv_combined.at[(idx, "X_adv_ssim"), col] = (
            "\\bfseries " + df_adv_combined.at[(idx, "X_adv_ssim"), col]
        )
    for col, idx in best_adv_psnr.iteritems():
        df_adv_combined.at[(idx, "X_adv_psnr"), col] = (
            "\\bfseries " + df_adv_combined.at[(idx, "X_adv_psnr"), col]
        )

    # rename rows and columns
    df_adv_combined = df_adv_combined.rename(
        index={
            "X_adv_err": "rel.~$\\l{2}$-err. [\\%]",
            "X_adv_ssim": "SSIM",
            "X_adv_psnr": "PSNR",
        }
    )
    df_adv_combined = df_adv_combined.rename(
        index=methods["info"].apply(lambda res: res["name_disp"]).to_dict()
    )

    # save latex tabular
    df_adv_combined.to_latex(
        os.path.join(save_path, "table_adv.tex"),
        column_format=2 * "l" + len(noise_rel) * "S[separate-uncertainty]",
        multirow=True,
        escape=False,
    )
