import os

import matplotlib as mpl
import torch
import numpy as np
from PIL import Image

from operators import (
    RadialMaskFunc,
)


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# ----- measurement configuration -----
mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.squeeze()
mask = np.uint8(mask.cpu().numpy())*255;

pim = Image.fromarray(mask)
pim.save('mask_stab_acc_experiment.png');

print('mask.shape: ', mask.shape )
print('mask.dtype: ', mask.dtype )
print('np.amax(mask): ', np.amax(mask) )




