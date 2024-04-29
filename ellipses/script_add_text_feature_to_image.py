# general imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat
import os

# personal imports
import config

savedir     = os.path.join(config.DATA_PATH,     "val")
savedir_toy = os.path.join(config.TOY_DATA_PATH, "val")
plotdir     = os.path.join(config.PLOT_PATH,     "add_text")

idx = "00042"
#idx = "00021"
#idx = "21"

fname = f'sample_{idx}.pt'
loadfn = os.path.join(savedir, fname)
#loadfn = os.path.join(savedir_toy, fname)
data = torch.load(loadfn);

print(data.dtype)

data = data.numpy()
print('np.amax(data): ', np.amax(data))

im = Image.fromarray(np.uint8(np.round(255*data)));

#fsize = 13;
fsize = 8;

draw = ImageDraw.Draw(im);
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', fsize);
text_intensity = 100 # CANCER detail
#text_intensity = 100 # smiley detail
draw.text((145,110), "CANCER", (text_intensity,), font=font);
#draw.text((125,125), "¯\_(ツ)_/¯", (text_intensity,), font=font);
#draw.text((152,140), "¯\_(ツ)_/¯", (text_intensity,), font=font);
im.save(os.path.join(plotdir, f'sample_N_256_{idx}.png') );
np_im = np.asarray(im, dtype='float32')/255;

new_data = torch.from_numpy(np_im);
print(new_data.dtype)
print("rel. l2-norm of the detail",  np.sum((np_im - data)**2)**.5 / np.sum(data**2)**.5 )
fname_out = f"sample_{idx}_text"
savefn = os.path.join(savedir, fname_out)
#savefn = os.path.join(savedir_toy, fname_out)
torch.save(new_data, "%s.pt"%savefn);
savemat(os.path.join(savedir, fname_out+'.mat'), {'data': np_im})
#savemat(os.path.join(savedir_toy, fname_out+'.mat'), {'data': np_im})




