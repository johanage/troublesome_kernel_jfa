# loss functions and training loop(s)
import torch; from torch import nn
from torch.nn.utils import spectral_norm
import pandas as pd
from typing import Tuple, Union
from networks import Generator, Discriminator
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
from operators import Fourier as Fourier

def plot_training_diagnostics(
    logging         : pd.DataFrame,
    image           : torch.Tensor,
    generated_image : torch.Tensor,
) -> None:
    fig, axs = plt.subplots(1,6, figsize = (25,5))
    axs[0].plot(logging["generator_loss"]);                          axs[0].set_title("Generator loss")
    axs[1].plot(logging["discriminator_loss"]);                      axs[1].set_title("Discriminator loss")
    axs[2].plot(logging["lr_generator"],     label="Generator");     axs[2].set_title("Learning schedule")
    axs[2].plot(logging["lr_discriminator"], label="Discriminator"); axs[2].legend()
    axs[3].plot(logging["mem_alloc"]);                               axs[3].set_title("Memory consumption")
    axs[4].imshow( (generated_image[0,0]**2 + generated_image[0,1]**2)**.5 )
    axs[4].set_title("Generated image")
    axs[5].imshow(image)
    axs[5].set_title("Example real image")
    fig.savefig(os.getcwd() + "/GAN_test_log.png")

def train_loop_gan(
    train_params          : dict,
    generator_params      : dict,
    generator             : Generator,
    discriminator         : Discriminator,
    data_load_train       : torch.utils.data.DataLoader,
    device                : torch.device,
    optimizer_G           : torch.optim.Adam,
    optimizer_D           : torch.optim.Adam,
    scheduler_G           : torch.optim.lr_scheduler.StepLR,
    scheduler_D           : torch.optim.lr_scheduler.StepLR,
    # generalize to all loss function modules?
    adversarial_loss_func : nn.modules.loss.BCELoss,
    logging               : pd.DataFrame,
    wgan                  : bool = False,
    ncritic               : int = 10,
    save_epochs           : int = 10,
    jitter                : bool = True,
    mag_jitter            : float = 0.1,
    fn_suffix             : str = ""
) -> Tuple[nn.Module, nn.Module, pd.DataFrame]:
    
    for epoch in range(train_params["num_epochs"]):
        # make sure we are in train mode
        generator.train()
        discriminator.train()
        progress_bar = tqdm(
            enumerate(data_load_train),
            desc="Train GAN epoch %i"%epoch,
            total=len(data_load_train.dataset)//train_params["batch_size"],
        )
        for i, batch in progress_bar:
            measurements, images = batch
            measurments = measurements.to(device); images = images.to(device)
            # add random noise
            if jitter:
                # gaussian noise
                images += mag_jitter * torch.randn(images.shape).cuda()
            
            # draw random latent vector z ~ N(0,I)
            latent_vector = torch.randn(images.shape[0], generator_params["latent_dim"]).to(device)
            #latent_vector = torch.rand(images.shape[0], generator_params["latent_dim"]).to(device)

            # ----------------------------------------------------------
            # Train discriminator
            # ----------------------------------------------------------
            optimizer_D.zero_grad()
            # measure discriminator's ability to distinguish real from generated images
            discriminated_imgs = discriminator.forward(images).detach()
            generated_images = generator.forward(latent_vector).to(device).detach()
            assert generated_images.shape == images.shape, "generated vector does not have the same shape as images - update your training parameters"
            discriminated_gen_imgs = discriminator.forward(generated_images).to(device)
            # TODO: implement WGAN discriminator loss
            if wgan:
                discriminator_loss = discriminated_imgs - discriminated_gen_imgs
            else:
                # equivalent with -log(D(x)), if BCE-loss
                real_loss = adversarial_loss_func( discriminated_imgs, torch.ones_like(discriminated_imgs))#, requires_grad=True) )
                # equivalent with -log( 1 - D( G(z) ) ), if BCE-loss
                fake_loss = adversarial_loss_func( discriminated_gen_imgs, torch.zeros_like(discriminated_gen_imgs))#, requires_grad=True) )
                # equivalent with -.5 * [ log( D(x) ) + log( 1 - D( G(z) ) ) ] 
                discriminator_loss = (real_loss + fake_loss)/2
            # backprop
            discriminator_loss.backward(retain_graph=True)
            optimizer_D.step()
            # update learning rate
            scheduler_D.step()
 
            # -----------------------------------------------------------
            # Train generator
            # -----------------------------------------------------------
            optimizer_G.zero_grad()
            # reduntant or required to reset computational graph?
            #generated_images = generator.forward(latent_vector).to(device)
            #discriminated_gen_imgs = discriminator.forward(generated_images).to(device)
            # TODO : implement WGAN generator loss
            if wgan:
                generator_loss = -discriminated_gen_imgs 
            else:
                # GAN generator loss equivalent with -log( D( G(z) ) ), if BCE-loss
                generator_loss = adversarial_loss_func(discriminated_gen_imgs, torch.ones_like(discriminated_gen_imgs) )        
            # backprop
            generator_loss.backward()
            optimizer_G.step()
            # update learning rate
            scheduler_G.step()           
            
            # ----------------------------------------------------------
            # LOGGING
            # ----------------------------------------------------------
            # append to log
            app_log = pd.DataFrame(
                {
                "generator_loss"     : generator_loss.item(),
                "discriminator_loss" : discriminator_loss.item(),
                "lr_generator"       : scheduler_G.get_last_lr()[0],
                "lr_discriminator"   : scheduler_D.get_last_lr()[0],
                "mem_alloc"          : torch.cuda.memory_allocated(),
                },
                index = [0] )
            logging = pd.concat([logging, app_log], ignore_index=True, sort=False)

            # update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(**{
                "gen_loss"   : generator_loss.item(),
                "discr_loss" : discriminator_loss.item(),
            }
            )
            # save generator and discriminator weights every save_epochs epochs
            if epoch % save_epochs == 0:
                # ----------------- SAVE G and D parameters ------------------------------------
                path = train_params["save_path"]
                # save generator weights and biases
                torch.save(generator.state_dict(), path + "/generator{suffix}_epoch{epoch}.pth".format(suffix = fn_suffix, epoch=epoch) )
                # save discriminator weights and biases
                torch.save(discriminator.state_dict(), path + "/discriminator{suffix}_epoch{epoch}.pth".format(suffix = fn_suffix, epoch=epoch))

                # ----------------- Plot diagnostics --------------------------------------------
                # image
                image = (data_load_train.dataset[0][1][0]**2 + data_load_train.dataset[0][1][1]**2)**.5
                # generate image
                z = torch.randn(1, generator_params["latent_dim"]).to(device)
                generated_image = generator.forward(z).to("cpu").detach()
                # plot
                plot_training_diagnostics(
                    logging         = logging,
                    image           = image,
                    generated_image = generated_image,
                )
    return (generator, discriminator, logging,)

# --------------------------------------------------------------------------------
#  Optimization of input
# --------------------------------------------------------------------------------
def regulariser_z(
    lmbda : float,
    z     : torch.Tensor,
    p     : Union[int,float] = 2,
) -> float:
    """
    From Bora et al. 2017 : Compressed sensing using Generative Models
    Eq. 3

    Args:
     - lmbda : regularisation parameter, measures the relative importance
               of the prior as compared to the measurement error
     - z     : random input vector for the generator
     - p     : degree of the norm used in the regularizer
    
    Out: the regularisation term
    """
    return lmbda * torch.norm(z.flatten(), p = p)**2

def objective_cs_gan(
    z         : torch.Tensor,
    y         : torch.Tensor,
    generator : Generator,
    OpA       : Fourier,
    regulate  : bool  = True,
    lmbda     : float = 1e-3,
    p_reg     : Union[int, float] = 2,
    p_meas    : Union[int, float] = 2,
) -> float:
    """
    Args:
     - z         : generator's random input vector 
     - y         : measurements
     - generator : pre-trained generator
     - OpA       : measurement operator, Fourier operator for MR-imaging
     - regulate  : include/exclude regularisation term lambda || R(z) ||^2
     - lmbda     : regularisation parameter, measures the relative importance
                   of the prior as compared to the measurement error
     - p_reg     : norm-degree for the regularisation term
     - p_meas    :         -- || --    measurement error term 
    """
    
    R_z = int(regulate) * regulariser_z(lmbda,z, p_reg)
    # generate image
    G_z = generator(z)
    # get approx measurements of generated image using Fourier operator
    y_tilde = OpA(G_z)
    # compute mesurement residuals
    res = y_tilde - y
    # compute measurement error according to norm of deg p_meas
    measurement_lperror = torch.norm(res.flatten(), p=p_meas)
    return measurement_lperror + R_z, measurement_lperror

from metrics import gan_representation_error

def optimize_generator_input_vector(
    cs_train_params      : dict,
    z_init               : torch.Tensor,
    generator            : Generator,
    data_load_train      : torch.utils.data.DataLoader,
    measurement_operator : Fourier,
    logging              : pd.DataFrame,
    device               : torch.device,
    save_epochs          : int = 10,
    fn_suffix            : str = "",
) -> Tuple[torch.Tensor, pd.DataFrame]:
    # set current random vector to initial random input vector
    z = z_init
    z.requires_grad_(True)
    z.retain_grad()
    optimizer = cs_train_params["optimizer"]([z],       **cs_train_params["optimizer_params"] )
    scheduler = cs_train_params["scheduler"](optimizer, **cs_train_params["scheduler_params"] )
    for epoch in range(cs_train_params["num_epochs"]):
        # make sure we are in train mode
        generator.eval()
        progress_bar = tqdm(
            enumerate(data_load_train),
            desc="Train GAN epoch %i"%epoch,
            total=len(data_load_train.dataset)//cs_train_params["batch_size"],
        )
        for i, batch in progress_bar:
            # get measurements and images from batch
            measurements, images = batch
            # make sure the datapairs are on the GPU
            measurements = measurements.to(device)
            images       = images.to(device)
            # set optimizer grads to zero
            optimizer.zero_grad()
            #breakpoint()
            loss_cs_gan, meas_err = objective_cs_gan(
                z         = z,
                y         = measurements,
                generator = generator,
                OpA       = measurement_operator,
            )
            loss_cs_gan.backward()
            meas_err.detach()
            optimizer.step()
            scheduler.step()
            G_z = generator(z)#.cuda()
            rep_err = gan_representation_error(G_z = G_z, x = images)
            rep_err.detach()
            # ----------------------------------------------------------
            # LOGGING
            # ----------------------------------------------------------
            # append to log
            app_log = pd.DataFrame(
                {
                "objective"            : loss_cs_gan.item(),
                "measurement_error"    : meas_err.item(),
                "representation_error" : rep_err.item(),
                "lr"                   : scheduler.get_last_lr()[0],
                "mem_alloc"            : torch.cuda.memory_allocated(),
                },
                index = [0] )
            logging = pd.concat([logging, app_log], ignore_index=True, sort=False)
            
            # update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(**{
                "objective"  : loss_cs_gan.item(),
                "meas_err"   : meas_err.item(),
                "rep_err"    : rep_err.item(),
            }
            )
             # save generator and discriminator weights every save_epochs epochs
            if epoch % save_epochs == 0:
                # ----------------- SAVE G and D parameters ------------------------------------
                path = cs_train_params["save_path"]
                # save input vector
                torch.save(z, path + "/z_optim{suffix}_epoch{epoch}.pt".format(suffix = fn_suffix, epoch=epoch))

                # ----------------- Plot diagnostics --------------------------------------------
                # plot
                fig, axs = plt.subplots(1,4, figsize=(20,5))
                axs[0].plot(logging["objective"], label="Objective")
                axs[0].plot(logging["measurement_error"], label="Measurement error")
                axs[0].plot(logging["representation_error"], label="Representation error")
                axs[1].plot(logging["lr"]); axs[1].set_title("Learning rate")
                image_img = images[0].detach().cpu()
                image = (image_img[0]**2 + image_img[1]**2)**.5
                gen_image_img = G_z[0].detach().cpu()
                gen_image = (gen_image_img[0]**2 + gen_image_img[0]**2)**.5
                axs[2].imshow(gen_image); axs[2].set_title("Generated image")
                axs[3].imshow(image); axs[2].set_title("Real image")
                fig.savefig(os.getcwd() + "/z_optim_and_diagnostics.png")

    return (z, logging,)
