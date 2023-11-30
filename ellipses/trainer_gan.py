# loss functions and training loop(s)
import torch; from torch import nn
import pandas as pd
from typing import Tuple
from networks import Generator, Discriminator
from tqdm import tqdm

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
    save_epochs           : int = 10,
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
            
            # -----------------------------------------------------------
            # Train generator
            # -----------------------------------------------------------
            optimizer_G.zero_grad()
            # draw random latent vector z
            latent_vector = torch.randn(images.shape[0], generator_params["latent_dim"]).to(device)
            #latent_vector = torch.rand(images.shape[0], generator_params["latent_dim"]).to(device)
            generated_images = generator.forward(latent_vector).to(device)
            assert generated_images.shape == images.shape, "generated vector does not have the same shape as images - update your training parameters"
            discriminated_gen_imgs = discriminator.forward(generated_images).to(device)
            # equivalent with -log( 1 - D( G(z) ) ), if BCE-loss
            generator_loss = adversarial_loss_func(discriminated_gen_imgs, torch.ones_like(discriminated_gen_imgs) )
            # equivalent with -log( D( G(z) ) ), if BCE-loss
            #generator_loss = adversarial_loss_func(discriminated_gen_imgs, torch.zeros_like(discriminated_gen_imgs) )
            generator_loss.backward(retain_graph=True)
            optimizer_G.step()
            scheduler_G.step()
            
            # ----------------------------------------------------------
            # Train discriminator
            # ----------------------------------------------------------
            optimizer_D.zero_grad()
            # measure discriminator's ability to distinguish real from generated images
            discriminated_imgs = discriminator.forward(images)
            generated_images = generator.forward(latent_vector).to(device)
            discriminated_gen_imgs = discriminator.forward(generated_images).to(device)
            # equivalent with -log(D(x)), if BCE-loss
            real_loss = adversarial_loss_func( discriminated_imgs, torch.ones_like(discriminated_imgs))#, requires_grad=True) )
            # equivalent with -log( 1 - D( G(z) ) ), if BCE-loss
            fake_loss = adversarial_loss_func( discriminated_gen_imgs, torch.zeros_like(discriminated_gen_imgs))#, requires_grad=True) )
            # equivalent with -.5 * [ log( D(x) ) + log( 1 - D( G(z) ) ) ] 
            discriminator_loss = (real_loss + fake_loss)/2
            discriminator_loss.backward()
            optimizer_D.step()
            scheduler_D.step()
            
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
                path = train_params["save_path"]
                # save generator weights and biases
                torch.save(generator.state_dict(), path + "/generator.pth")
                # save discriminator weights and biases
                torch.save(discriminator.state_dict(), path + "/discriminator.pth")
    return (generator, discriminator, logging,)
