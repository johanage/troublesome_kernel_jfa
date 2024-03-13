# plot gt images from toy and case study dataset

# external import
import torch, os
from matplotlib import pyplot as plt

# turn of axis and remove margins
rc = {"axes.spines.left"   : False,
      "axes.spines.right"  : False,
      "axes.spines.bottom" : False,
      "axes.spines.top"    : False,
      "xtick.bottom"       : False,
      "xtick.labelbottom"  : False,
      "ytick.labelleft"    : False,
      "ytick.left"         : False,
      "axes.xmargin"       : 0,
      "axes.ymargin"       : 0,
      "savefig.bbox"       : "tight"
}

plt.rcParams.update(rc)

# personal imports
import config

# load and plot ellipses sample 0 to 3
ellipses_examples = {}
for i in range(4):
    ellipses_examples[i] = torch.load(os.path.join(config.TOY_DATA_PATH, "train", "sample_%i.pt"%i))
    plt.figure()
    plt.imshow(ellipses_examples[i], cmap="Greys_r")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "plots", "gt_examples", "ellipses_%i"%i), bbox_inches="tight", pad_inches=0)
    plt.clf()
# load and plot brain sample 0 to 3
brain_examples = {}
for i in range(4):
    brain_examples[i] = torch.load(os.path.join(config.DATA_PATH, "train", "sample_0000%i.pt"%i))
    plt.figure()
    plt.imshow(brain_examples[i], cmap="Greys_r")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "plots", "gt_examples", "brain_%i"%i), bbox_inches="tight", pad_inches=0)
    plt.clf()
