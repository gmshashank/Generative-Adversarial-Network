import argparse
import glob
import os
from collections import OrderedDict
from logging import root
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb

# from torch.utils.data.dataset ilsmport Dataset
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

MANUAL_SEED = 42


class Generator(nn.Module):
    """
    Creates the Generator

    nz (int): size of the latent z vector
    ngf (int): number of feature maps for the generator
    """

    def __init__(self, latent_dim: int = 64, ngf: int = 64, nc: int = 3):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    Creates the Discriminator

    nc (int): number of channels of the input image
    ndf (int): number of feature maps for the discriminator

    This uses the special Spectral Normalization ref: https://arxiv.org/abs/1802.05957
    """

    def __init__(self, ndf: int = 64, nc: int = 3):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            torch.nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            torch.nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            torch.nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.utils.spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    """ 
    Custom weights initialization called on Generator and Discriminator
    Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _download_and_process_data():
    import tarfile

    from torchvision.datasets.utils import download_url

    # Dowload the train dataset
    # dataset_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
    dataset_url = "http://ai.stanford.edu/~jkrause/car196/car_ims.tgz"
    download_url(dataset_url, ".")

    # Extract from archive
    with tarfile.open("./car_ims.tgz", "r:gz") as tar:
        tar.extractall(path="./data/")

    # Download DevKit https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
    devkit_dataset_url = "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
    download_url(devkit_dataset_url, ".")

    # Extract from archive
    with tarfile.open("./car_devkit.tgz", "r:gz") as tar:
        tar.extractall(path="./data/devkit")


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        num_workers: int = 2,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
        data_path: str = "./data/",
        num_epochs=20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.fixed_noise = torch.randn(32, self.latent_dim, 1, 1)

        self.generator = Generator(latent_dim=self.latent_dim, ngf=self.ngf, nc=self.nc).apply(weights_init)
        self.discriminator = Discriminator(ndf=self.ndf, nc=self.nc).apply(weights_init)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        noise = torch.randn(self.batch_size, self.latent_dim, 1, 1).to(self.device)
        fake = self.generator(noise)
        # print(real.shape,noise.shape,fake.shape)

        # train generator
        if optimizer_idx == 0:
            output = self.discriminator(fake).reshape(-1)
            loss_gen = self.adversarial_loss(output, torch.ones_like(output))
            self.log("train/g_loss", loss_gen)
            # print(f"loss_gen: {loss_gen}")
            return loss_gen

        # train discriminator
        if optimizer_idx == 1:
            disc_real = self.discriminator(real).reshape(-1)
            loss_disc_real = self.adversarial_loss(disc_real, torch.ones_like(disc_real))
            disc_fake = self.discriminator(fake.detach()).reshape(-1)
            loss_disc_fake = self.adversarial_loss(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            self.log("train/d_loss", loss_disc)
            # print(f"loss_disc: {loss_disc}")
            return loss_disc

    def validation_step(self, batch, batch_idx):
        real, _ = batch
        noise = self.fixed_noise.to(self.device)
        fake = self.generator(noise)

        img_grid_real = torchvision.utils.make_grid(real, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        self.logger.experiment.add_image("Real", img_grid_real, self.current_epoch)
        self.logger.experiment.add_image("Fake", img_grid_fake, self.current_epoch)

    def configure_optimizers(self):
        # print("configure_optimizers")
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [optimizer_g, optimizer_d], []

    def setup(self, stage: str):
        # print("setup")
        data_dir = self.data_path
        mean = [0.570838093757629, 0.479552984237671, 0.491760671138763]
        std = [0.279659748077393, 0.309973508119583, 0.311098515987396]

        transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean, std, inplace=True),]
        )
        original_dataset = torchvision.datasets.ImageFolder(data_dir, transform)

        split_a_size = int(0.8 * len(original_dataset))
        split_b_size = len(original_dataset) - split_a_size
        train, valid = torch.utils.data.random_split(
            original_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(MANUAL_SEED),
        )

        # assign to use in dataloaders
        self.train_dataset = train
        self.val_dataset = valid

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def on_epoch_end(self):
        noise = self.fixed_noise.to(self.device)
        fake = self.generator(noise)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        self.logger.experiment.add_image("generated_images", img_grid_fake, self.current_epoch)


def main(args: argparse.Namespace) -> None:

    # Init model from datamodule's attributes
    model = DCGAN(**vars(args))

    logger = pl.loggers.TensorBoardLogger(save_dir="/logs")
    args.weight_summary = "full"

    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    callbacks = []
    args.weights_summary = "full"  # Print full summary of the model

    # Init trainer
    if args.gpus:
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.num_epochs,
            callbacks=callbacks,
            logger=logger,
            weights_save_path="training/logs",
        )
    elif args.tpu_cores:
        trainer = pl.Trainer(
            tpu_cores=args.tpu_cores,
            max_epochs=args.num_epochs,
            callbacks=callbacks,
            logger=logger,
            weights_save_path="training/logs",
        )

    trainer.tune(model)  # If passing --auto_lr_find, this will set learning rate

    # Train
    trainer.fit(model)
    trainer.save_checkpoint(f"carGAN_custom_checkpoint_{args.num_epochs}.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--tpu_cores", type=int, default=0, help="number of TPUs")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="number of worker threads")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to Image folder")
    parser.add_argument("--batch_size", type=int, default=64, help="size of batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam: learning rate")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2", type=float, default=0.999, help="Adam: decay of first order momentum of gradient",
    )
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of latent space")
    args = parser.parse_args()
    main(args)
