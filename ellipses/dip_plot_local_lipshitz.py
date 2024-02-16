# general imports
import os, torch
from matplotlib import pyplot as plt
# loacl imports
import config
epochs = torch.arange(80)*100
breakpoint()
# load no jit DIP image reconstructions
rec_dip_nojit = [torch.load(os.path.join(config.RESULTS_PATH, "DIP", "DIP_nojit_rec_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1_epoch{epoch}.pt".format(epoch=epoch))) for epoch in epochs]

# load jit DIP image reconstructions
rec_dip_jit = [torch.load(os.path.join(config.RESULTS_PATH, "DIP", "DIP_jit_eta_0.1_rec_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1_epoch{epoch}.pt".format(epoch=epoch))) for epoch in epochs]

# load perturbations
perturbation = torch.load(os.path.join(config.RESULTS_PATH, "DIP", "DIP_jit_eta_0.1_perturbation_lr_0.0005_gamma_0.96_sp_circ_sr2.5e-1.pt"))
L = torch.zeros((len(epochs),))
for i, (fx, fy) in enumerate(zip(rec_dip_nojit, rec_dip_jit)):
    L[i] = (fy - fx).norm(p=2) / perturbation.norm(p=2)

plt.plot(list(epochs), L)
plt.savefig(os.path.join(config.RESULTS_PATH, "../plots/DIP/local_lipshitz.png"))
