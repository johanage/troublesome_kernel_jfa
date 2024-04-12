import os
from data_management import sample_ellipses

N = 256;
ROOT_TUXEDO           = "/home/johanfag/cs"
ROOT_IFI              = "/itf-fi-ml/home/johanfag"
ROOT                  = "/uio/kant/geo-exports-u1/johanfag"
TOY_DATA_PATH         = os.path.join(ROOT, "master/codebase/data/ellipses")
DATA_PATH             = os.path.join(ROOT, "master/codebase/data/pytorch_datasets/fastMRI")
RESULTS_PATH          = os.path.join(ROOT, "master/codebase/troublesome_kernel_jfa/ellipses/models")
RESULTS_PATH_KADINGIR = "/mn/kadingir/afha_000000/nobackup/JohanAgerup/models"
SCRATCH_PATH          = os.path.join("/mn/nam-shub-02/scratch/johanfag")
# sampling pattern path
SP_PATH    = os.path.join(ROOT, "master/codebase/troublesome_kernel_jfa/ellipses/sampling_patterns")
# plot path
PLOT_PATH  = os.path.join(ROOT, "master", "codebase", "troublesome_kernel_jfa", "ellipses", "plots")     

# ----- random seeds -----
torch_seed = 1
numpy_seed = 2
matrix_seed = 3

# Sampling pattern
use_pattern_from_file = False
#fname_patt = f'/mn/sarpanitu/ansatte-u4/vegarant/storage_stable_NN/samp_patt/XXX.png'

# ----- signal configuration -----
n = (N, N)  # signal dimension
data_params = {  # additional data generation parameters
    "c_min": 10,
    "c_max": 40,
    "max_axis": 0.15,
    "min_axis": 0.01,
    "margin_offset": 0.3,
    "margin_offset_axis": 0.9,
    "grad_fac": 0.9,
    "bias_fac": 1.0,
    "bias_fac_min": 0.3,
    "normalize": True,
}
data_gen = sample_ellipses  # data generator function

# ----- data set configuration -----
set_params = {
    "num_train": 25000,
    "num_val": 1000,
    "num_test": 1000,
    "path": DATA_PATH,
}
