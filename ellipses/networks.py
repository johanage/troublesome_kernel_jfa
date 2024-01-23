import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import pandas as pd
import torch

from tqdm import tqdm

from operators import l2_error, to_complex


# ----- ----- Abstract Base Network ----- -----


class InvNet(torch.nn.Module, metaclass=ABCMeta):
    """ Abstract base class for networks solving linear inverse problems.

    The network is intended for the denoising of a direct inversion of a 2D
    signal from (noisy) linear measurements. The measurement model

        y = Ax + noise

    can be used to obtain an approximate reconstruction x_ from y using, e.g.,
    the pseudo-inverse of A. The task of the network is either to directly
    obtain x from y or denoise and improve this first inversion x_ towards x.

    """

    def __init__(self):
        super(InvNet, self).__init__()

    @abstractmethod
    def forward(self, z):
        """ Applies the network to a batch of inputs z. """
        pass

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_step(
        self, batch_idx, batch, loss_func, optimizer, batch_size, acc_steps
    ):
        inp, tar = batch
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        pred = self.forward(inp)

        loss = loss_func(pred, tar) / acc_steps
        loss.backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, pred

    def _val_step(self, batch_idx, batch, loss_func):
        inp, tar = batch
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        pred = self.forward(inp)
        loss = loss_func(pred, tar)
        return loss, inp, tar, pred

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        inp,
        tar,
        pred,
        v_loss,
        v_inp,
        v_tar,
        v_pred,
        val_data,
    ):

        self._print_info()
        # DataFrame.append: Deprecated since version 1.4.0: Use concat() instead. 
        #logging = logging.append(
        app_log = pd.DataFrame(
            {
                "loss": loss.item(),
                "val_loss": v_loss.item(),
                "rel_l2_error": l2_error(
                    pred, to_complex(tar), relative=True, squared=False
                )[0].item(),
                "val_rel_l2_error": l2_error(
                    v_pred, to_complex(v_tar), relative=True, squared=False
                )[0].item(),
            }, 
            index = [0]
        ) 
        logging = pd.concat(
            [logging, app_log],
            ignore_index=True,
            sort=False,
        )

        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
            )
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{}.pt".format(epoch + 1)
                ),
            )
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{}.pkl".format(epoch + 1)
                ),
            )
            fig.savefig(
                os.path.join(save_path, "plot_epoch{:03d}.png".format(epoch + 1)),
                bbox_inches="tight",
            )
            del fig

        return logging

    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):
        """ Can be overwritten by child classes to plot training progress. """
        pass

    def _add_to_progress_bar(self, dict):
        """ Can be overwritten by child classes to add to progress bar. """
        return dict

    def _on_train_end(self, save_path, logging):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "model_weights.pt")
        )
        logging.to_pickle(os.path.join(save_path, "losses.pkl"),)

    def _print_info(self):
        """ Can be overwritten by child classes to print at epoch end. """
        pass

    def train_on(
        self,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        loss_func,
        save_path,
        save_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2e-4, "eps": 1e-3},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 1.0},
        acc_steps=1,
        train_transform=None,
        val_transform=None,
        train_loader_params={"shuffle": True},
        val_loader_params={"shuffle": False},
        plot_evolution = True,
        fn_evolution = None,
    ):
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)

        train_data.transform = train_transform
        val_data.transform = val_transform
        train_loader_params = dict(train_loader_params)
        val_loader_params = dict(val_loader_params)
        if "sampler" in train_loader_params:
            train_loader_params["sampler"] = train_loader_params["sampler"](
                train_data
            )
        if "sampler" in val_loader_params:
            val_loader_params["sampler"] = val_loader_params["sampler"](
                val_data
            )

        data_load_train = torch.utils.data.DataLoader(
            train_data, batch_size, **train_loader_params
        )
        data_load_val = torch.utils.data.DataLoader(
            val_data, batch_size, **val_loader_params
        )

        logging = pd.DataFrame(
            columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
        )
        if plot_evolution:
            fig_evo, axs_evo = plt.subplots(2,10,figsize=(50,10) )
            isave = 0
            [ax.set_axis_off() for ax in axs_evo.flatten()]
        for epoch in range(num_epochs):
            self.train()  # make sure we are in train mode
            t = tqdm(
                enumerate(data_load_train),
                desc="epoch {} / {}".format(epoch, num_epochs),
                total=-(-len(train_data) // batch_size),
            )
            optimizer.zero_grad()
            loss = 0.0
            for i, batch in t:
                loss_b, inp, tar, pred = self._train_step(
                    i, batch, loss_func, optimizer, batch_size, acc_steps
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                with torch.no_grad():
                    loss += loss_b
                    loss /= i + 1

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()
                v_loss = 0.0
                for i, v_batch in enumerate(data_load_val):
                    v_loss_b, v_inp, v_tar, v_pred = self._val_step(
                        i, v_batch, loss_func
                    )
                    v_loss += v_loss_b
                v_loss /= i + 1
                
                # logging
                logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    inp,
                    tar,
                    pred,
                    v_loss,
                    v_inp,
                    v_tar,
                    v_pred,
                    val_data,
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"val_rel_l2_err": logging["val_rel_l2_error"].values[-1] })
                )
        self._on_train_end(save_path, logging)
        return logging

# ----- ----- U-Net ----- -----

class UNet(InvNet):
    """ U-Net implementation.

    Based on https://github.com/mateuszbuda/brain-segmentation-pytorch/
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self, in_channels=1, out_channels=1, base_features=32, drop_factor=0.0, inverter = None, operator = None,
    ):
        # set properties of UNet
        super(UNet, self).__init__()
        self.inverter = inverter
        self.operator = operator
        self.encoder1 = UNet._conv_block(
            in_channels,
            base_features,
            drop_factor=drop_factor,
            block_name="encoding_1",
            device = self.device,
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._conv_block(
            base_features,
            base_features * 2,
            drop_factor=drop_factor,
            block_name="encoding_2",
            device = self.device,
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._conv_block(
            base_features * 2,
            base_features * 4,
            drop_factor=drop_factor,
            block_name="encoding_3",
            device = self.device,
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._conv_block(
            base_features * 4,
            base_features * 8,
            drop_factor=drop_factor,
            block_name="encoding_4",
            device = self.device,
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._conv_block(
            base_features * 8,
            base_features * 16,
            drop_factor=drop_factor,
            block_name="bottleneck",
            device = self.device,
        )
        
        self.upconv4 = torch.nn.ConvTranspose2d(
            base_features * 16, base_features * 8, kernel_size=2, stride=2,
        )
        self.upconv4.to(self.device)
        self.decoder4 = UNet._conv_block(
            base_features * 16,
            base_features * 8,
            drop_factor=drop_factor,
            block_name="decoding_4",
            device = self.device,
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            base_features * 8, base_features * 4, kernel_size=2, stride=2
        )
        self.upconv3.to(self.device)
        self.decoder3 = UNet._conv_block(
            base_features * 8,
            base_features * 4,
            drop_factor=drop_factor,
            block_name="decoding_3",
            device = self.device,
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_features * 4, base_features * 2, kernel_size=2, stride=2
        )
        self.upconv2.to(self.device)
        self.decoder2 = UNet._conv_block(
            base_features * 4,
            base_features * 2,
            drop_factor=drop_factor,
            block_name="decoding_2",
            device = self.device,
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=2, stride=2
        )
        self.upconv1.to(self.device)
        self.decoder1 = UNet._conv_block(
            base_features * 2,
            base_features,
            drop_factor=drop_factor,
            block_name="decoding_1",
            device = self.device,
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.outconv.to(self.device)

    def forward(self, x):
        # In the original code this was done in ItNet step
        if self.inverter is not None:
            x = self.inverter(x)
        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        # add enc4
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        # add enc3
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        # add enc2
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        # add enc1
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        breakpoint()
        return self.outconv(dec1)

    # TODO: add option to do upsampling with UpSampling and not just transpose convolution
    # NOTE: according to Ulyanov et al. transpose convolutions give worse results than Upsampling, see https://github.com/DmitryUlyanov/deep-image-prior/blob/master/models/unet.py
    
    @staticmethod
    def _conv_block(in_channels, out_channels, drop_factor, block_name, device = None, act_func = "leakyrelu", negative_slope = 0.1):
        if act_func == "leakyrelu":
            activation_function = torch.nn.LeakyReLU(negative_slope=negative_slope)#, inplace=True)
        if act_func == "relu":
            activation_function = torch.nn.ReLU()#inplace=True)
        block = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (block_name + "bn_1", torch.nn.BatchNorm2d(out_channels)),
                    (block_name + act_func + "1", activation_function),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (block_name + "bn_2", torch.nn.BatchNorm2d(out_channels)),
                    (block_name + act_func + "2", activation_function),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )
        if device is not None:
            block.to(device)
        return block
    
    # plot function
    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):
        def _implot(sub, im):
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu(),
                    cmap = "Greys_r",                    
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu())
            return p

        fig, subs = plt.subplots(2, 3, clear=True, num=1, figsize=(15, 10))

        # inv = self.inverter(inp)
        v_inv = self.inverter(v_inp)

        # training and validation loss
        subs[0, 0].set_title("Losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()
   
        # validation input
        p10 = _implot(subs[1, 0], v_inv)
        subs[1, 0].set_title("val inv")

        # validation output
        p01 = _implot(subs[0, 1], v_pred)
        subs[0, 1].set_title(
            "val output:\n ||x0-xrec||_2 / ||x0||_2 \n = "
            "{:1.2e}".format(logging["val_rel_l2_error"].iloc[-1])
        )

        # validation target
        p11 = _implot(subs[1, 1], v_tar)
        subs[1, 1].set_title("val target")

        # validation difference
        p12 = _implot(subs[1, 2], v_pred - v_tar)
        subs[1, 2].set_title("val diff: x0 - x_pred")

        # training output
        p02 = _implot(subs[0, 2], pred)
        subs[0, 2].set_title(
            "train output:\n ||x0-xrec||_2 / ||x0||_2 \n = "
            "{:1.2e}".format(logging["rel_l2_error"].iloc[-1])
        )
        for ax,plot in zip([subs[1,0], subs[0,1], subs[1,1], subs[1,2], subs[0,2]],[p10, p01, p11, p12, p02]):
            divider = mal(ax)
            cax     = divider.append_axes("right", size="5%", pad = 0.05)
            fig.colorbar(plot, cax=cax)
        return fig

from torch import nn
class Generator(nn.Module):
    def __init__(
        self, 
        img_size        : int = 512, 
        latent_dim      : int = 100, 
        channels        : int = 1,
        num_upsample    : int = 2,
        factor_upsample : int = 2,
    ) -> None:
        """
        Args:
         - self       : torch nn module of class Generator
         - img_size   : image size
         - latent_dim : dimension of latent variable input, i.e. input to generator
         - channels   : channels of the output of the generator, 
                        typically we generate images with 3 or 1 channel
        """
        super(Generator, self).__init__()
        # mathces upsampling that happens two times with a factor of 2
        self.init_size = img_size // (num_upsample * factor_upsample) 
        # initial FC layer: (N, latent dimension) -> (N, 128 * image size ** 2)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        # Note input is of shape (N, 128, init_size, init_size), init_size = image size // 4
        self.conv_blocks = nn.Sequential(
            # block 1  
            nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2, inplace=True),
            # upsample : (N, 128, init_size, init_size) -> (N, 128, 2*init_size, 2*init_size)
            nn.Upsample(scale_factor=factor_upsample),
            # shape stays the same
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            # block 2 
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2),# inplace=True),
            # upsample : (N, 128, 2*init_size, 2*init_size) -> (N, 128, 4*init_size, 4*init_size)
            nn.Upsample(scale_factor=factor_upsample),
            # reshape channels C : 128 -> 64
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            # block 3 : outblock
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),# inplace=True),
            # reshape channels C : 64 -> channels
            nn.Conv2d(in_channels = 64, out_channels = channels, kernel_size = 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, 
        z : torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
         - z : the latent input variable
        """
        # linear layer : (N, latent dimension) -> (N, 128 * (init size)^2)
        out = self.l1(z)
        # NOTE: out.shape[0] = N
        # reshape : (N, 128 * (init size)^2) -> (N, 128, init size, init size) 
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # reshape : (N, 128, init size, init size) -> (N, channels, image size, image size) 
        img = self.conv_blocks(out)
        return img

from torch.nn.utils.parametrizations import spectral_norm
class Discriminator(nn.Module):
    def __init__(
        self, 
        img_size : int = 512, 
        channels : int = 1,
    ) -> None:
        """
        Args:
         - img_size : image size
         - chanels  : input channel size of the discriminator
                      that has to match output channel of the generator
        """
        super(Discriminator, self).__init__()

        def discriminator_block(
            in_filters  : int, 
            out_filters : int, 
            bn          : bool = True,
        ) -> list:
            """
            Args:
             - in_filters  : input filter size for the discriminator 2D convolutional blocks
             - out_filters : output filter size -- || --
             - bn          : use batch norm (bn) or not
            
            Out: list of torch.nn.Modules
            """
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1) ), 
                     nn.LeakyReLU(0.2), #inplace=True), 
                     nn.Dropout2d(0.25)
                    ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),# name="kernel" ),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            spectral_norm( nn.Linear(128 * ds_size ** 2, 1) ), 
            nn.Sigmoid())

    def forward(self, 
        img : torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
         - img : image input from generator
        Out: validity, tensor of values between 0 and 1 that indicates the 
             validity of the image, i.e. whether the image is fake or not
        """
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
