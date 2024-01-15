# optimize generator input z
import os
import matplotlib as mpl
import torch, torchvision

# vegard's implementations
from data_management import ToComplex, SimulateMeasurements
from networks import Generator, Discriminator
from operators import Fourier as Fourier
from operators import (
    RadialMaskFunc,
)


# ----- load configuration -----
import config

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cpu")
# if GPU availablei
gpu_avail = torch.cuda.is_available()
if gpu_avail:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

# ----- measurement configuration -----
mask_func = RadialMaskFunc(config.n, 40)
mask = mask_func((1,) + config.n + (1,))
mask = mask.squeeze(-1)
mask = mask.unsqueeze(1)
# Fourier operator
OpA = Fourier(mask)



from importlib import reload
import trainer_gan; reload(trainer_gan)
from trainer_gan import *
from dataclass import asdict
# logging
logging_zoptim = pd.DataFrame(
        columns=["objective", "measurement_error", "representation_error", "lr", "mem_alloc"]
)

# init cs gan params
cs_gan_train_params = asdict( CS_GAN_train_params() )

# init generator input
init_shape = (cs_gan_train_params["batch_size"], generator_params["latent_dim"])
z_init = torch.randn(init_shape).to(device)

# optimize input
z_optim, logging = optimize_generator_input_vector(
    cs_train_params      = cs_gan_train_params,
    z_init               = z_init,
    generator            = generator,
    data_load_train      = data_load_train,
    measurement_operator = OpA,
    logging              = logging_zoptim,
    device               = device,
)
