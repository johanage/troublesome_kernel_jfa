# general imports
from functools import partial
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat
import os

# personal imports
import config
from networks import UNet
from operators import (
    Fourier,
    Fourier_matrix as Fourier_m,
    MaskFromFile,
    LearnableInverterFourier,
    proj_l2_ball,
    to_complex,
)
from find_adversarial import PAdam

# ----- global configuration -----
device = torch.device("cpu")
# if GPU available
gpu_avail = torch.cuda.is_available()
if gpu_avail:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
 
# set directories
savedir     = os.path.join(config.DATA_PATH, "train")
savedir_toy = os.path.join(config.TOY_DATA_PATH, "train")
plotdir     = os.path.join(config.PLOT_PATH, "detail_transfer")

idx = "00042"

fname = f'sample_{idx}.pt'
loadfn = os.path.join(savedir, fname)
data = torch.load(loadfn);
data_det = torch.zeros_like(data)
print(data.dtype)

data = data.numpy()
data_det = data_det.numpy()
print('np.amax(data): ', np.amax(data))
print('np.amax(np.uint8(np.round(255*data))): ', np.amax(np.uint8(np.round(255*data))))

im           = Image.fromarray(np.uint8(np.round(255*data)));
im.save(os.path.join(plotdir, "original_image.png"))
im_det       = Image.fromarray(np.uint8(np.round(255*data_det)));
im_large_det = Image.fromarray(np.uint8(np.round(255*data_det)));

# ----- measurement configuration -----
sampling_rate = 0.25
print("sampling rate used is :", sampling_rate)
sp_type = "circle" # "diamond", "radial"
mask_fromfile = MaskFromFile(
    path = os.path.join(config.SP_PATH, "circle"), # circular pattern
    filename = "multilevel_sampling_pattern_%s_sr%.2f_a1_r0_2_levels50.png"%(sp_type, sampling_rate) # sampling_rate *100 % sr, a = 1, r0 = 2, nlevles = 50 
)
mask = mask_fromfile.mask[None]
# Fourier operator
OpA = Fourier(mask)

# ------- Construct the detail x_det s.t. || A x_det || << 1 -----------------------------------
draw = ImageDraw.Draw(im_det);
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 25);
text_intensity = 100
draw.text((100,150), "◉_◉", (text_intensity,), font=font);
np_imdet = np.asarray(im_det, dtype='float32')/255;
# detail as tensor
new_data_det = torch.from_numpy(np_imdet)
new_data_det_cmplx = to_complex(new_data_det[None,None])
# nullspace detail as tensor: x_det_nullspace = x_det - OpA^*(OpA(x_det))
nullspace_data_det = new_data_det_cmplx - OpA.adj(OpA( new_data_det_cmplx))
# save nullspace detail tensor
torch.save(nullspace_data_det, os.path.join(plotdir, "detail.pt") )
# make numpy array of nullsapce detail
np_imdet_nullspace = nullspace_data_det.norm(p=2,dim=(0,1)).numpy()
# make PIL image of nullspace detail
im_nullspace_det = Image.fromarray(np.round(255*np_imdet_nullspace).astype('uint8'))
# save to confirm that PIL image is correct
im_nullspace_det.save(os.path.join(plotdir, "nullspace_detail_pil-image.png"))

# save detail, and its orthogonal and parallel component
save_image(new_data_det, os.path.join(plotdir, "detail.png") )
save_image(OpA.adj(OpA( new_data_det_cmplx)).norm(p=2, dim=(0,1)), os.path.join(plotdir, "orth_detail.png") )
save_image(nullspace_data_det.norm(p=2, dim=(0,1)), os.path.join(plotdir, "nullspace_detail.png") )

# ------- Construct another detail x_det' s.t. ||A x_det'|| >> 1 --------------------------------
draw = ImageDraw.Draw(im_large_det);
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 25);
text_intensity = 90
draw.text((100,50), "(¬‿¬)", (text_intensity,), font=font);
np_im_large_det = np.asarray(im_large_det, dtype='float32')/255;
# detail as tensor
new_data_large_det = torch.from_numpy(np_im_large_det)
# save tensor and image
torch.save(new_data_large_det, os.path.join(plotdir, "large_detail.pt"))
torch.save(new_data_large_det + torch.tensor(data), os.path.join(plotdir, "image_plus_large_detail.pt"))
save_image(new_data_large_det, os.path.join(plotdir, "large_detail.png") )

# ------- Draw the detail x_det in the image x --------------------------------------------------
sum_im_details = data + np_imdet_nullspace + np_im_large_det
print("np.amax(sum_im_details)", np.amax(sum_im_details))
np_detailed_image = np.round( (255*sum_im_details)).astype('uint8')
print('np.amax(np_detailed_image): ', np.amax(np_detailed_image))

detailed_image = Image.fromarray(np_detailed_image) 
detailed_image.save(os.path.join(plotdir, f'sample_nullspace_detail_N_256_{idx}.png') );
np_im = np.asarray(detailed_image, dtype='float32')/255;
new_data = torch.from_numpy(np_im);
print(new_data.dtype)

# ------- Save the detailed image: x + x_det + x_large_det --------------------------------------------------
tensor_det_img = torch.tensor(data) + nullspace_data_det + new_data_large_det 
fname_out = f"sample_{idx}_neigh_nullspace_det"
savefn = os.path.join(savedir, fname_out)
torch.save(tensor_det_img[0], "%s.pt"%savefn);
savemat(os.path.join(savedir, fname_out+'.mat'), {'data': new_data})
