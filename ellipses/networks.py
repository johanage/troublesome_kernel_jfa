import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
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
        #breakpoint()
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
            #breakpoint()
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
        #breakpoint()
        if self.inverter is not None:
            x = self.inverter(x)
        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    @staticmethod
    def _conv_block(in_channels, out_channels, drop_factor, block_name, device = None):
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
                    (block_name + "relu1", torch.nn.ReLU(True)),
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
                    (block_name + "relu2", torch.nn.ReLU(True)),
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
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu()
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu())
            return p

        fig, subs = plt.subplots(2, 3, clear=True, num=1, figsize=(8, 6))

        # inv = self.inverter(inp)
        v_inv = self.inverter(v_inp)

        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()

        # validation input
        p10 = _implot(subs[1, 0], v_inv)
        subs[1, 0].set_title("val inv")
        plt.colorbar(p10, ax=subs[1, 0])

        # validation output
        p01 = _implot(subs[0, 1], v_pred)
        subs[0, 1].set_title(
            "val output:\n ||x0-xrec||_2 / ||x0||_2 \n = "
            "{:1.2e}".format(logging["val_rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p01, ax=subs[0, 1])

        # validation target
        p11 = _implot(subs[1, 1], v_tar)
        subs[1, 1].set_title("val target")
        plt.colorbar(p11, ax=subs[1, 1])

        # validation difference
        p12 = _implot(subs[1, 2], v_pred - v_tar)
        subs[1, 2].set_title("val diff: x0 - x_pred")
        plt.colorbar(p12, ax=subs[1, 2])

        # training output
        p02 = _implot(subs[0, 2], pred)
        subs[0, 2].set_title(
            "train output:\n ||x0-xrec||_2 / ||x0||_2 \n = "
            "{:1.2e}".format(logging["rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p02, ax=subs[0, 2])

        return fig
