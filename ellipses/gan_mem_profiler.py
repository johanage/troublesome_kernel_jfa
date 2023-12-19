"""
DESCRIPTION OF WHAT THIS SCRIPT DOES
------------------------------------
Terms
    - jitter, means that noise, typically Gaussian, is added to the data while training to reduce overfit
"""


import os, sys

import matplotlib as mpl
import torch
import torchvision

# vegard's implementations
from data_management import IPDataset, ToComplex, SimulateMeasurements
from networks import Generator, Discriminator
from operators import Fourier as Fourier
from operators import Fourier_matrix as Fourier_m
from operators import (
    RadialMaskFunc,
)


# ----- load configuration -----
import config  # isort:skip

# Enable memory profiler
from torch.profiler import profile, ProfilerActivity, record_function
with profile(
    activities=[ProfilerActivity.CUDA], 
    profile_memory=True, 
    record_shapes=True
) as prof:
    exec(open(os.getcwd() + str(sys.argv[1]) ).read() )
print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
