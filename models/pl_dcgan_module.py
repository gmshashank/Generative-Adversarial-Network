import argparse
from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from collections import OrderedDict

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .gan import Generator, Discriminator


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        beta1: float = 0.5,
        beta2: float = 0.999,
        feature_maps_generator: int = 64,
        feature_maps_discriminator: int = 64,
        image_channels: int = 3,
        latent_dim: int = 64,
        learning_rate: float = 0.0002,
        generator_loss_collection: List = [],
        discriminator_loss_collection: List = [],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()
        self.criterion = nn.BCELoss()
        self.generator_loss_collection = generator_loss_collection
        self.discriminator_loss_collection = discriminator_loss_collection

    @staticmethod
    def _weights_init(m):
        # custom weights initialization called on netG and netD
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def _get_generator(self) -> nn.Module:
        generator = Generator(self.hparams.latent_dim, self.hparams.feature_maps_generator, self.hparams.image_channels)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self) -> nn.Module:
        discriminator = Discriminator(self.hparams.feature_maps_discriminator, self.hparams.image_channels)
        discriminator.apply(self._weights_init)
        return discriminator

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr - lr, betas=betas)
        return [opt_gen, opt_disc], []

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = torch.rand(*noise.shape, 1, 1).to(self.device)
        return self.generator(noise)

    def _get_generator_loss(self, real: torch.Tensor) -> torch.Tensor:
        fake_prediction = self._get_fake_prediction(real)
        fake_ground_truth = torch.ones_like(fake_prediction)
        generator_loss = self.criterion(fake_prediction, fake_ground_truth)
        return generator_loss

    def _generator_step(self, real: torch.Tensor) -> torch.Tensor:
        generator_loss = self._get_generator_loss(real)
        self.log("loss/generator", generator_loss, on_epoch=True)
        return generator_loss
        # tqdm_dict = {"generator_loss",generator_loss}
        # generator_output = OrderedDict(
        #     {"loss": generator_loss, "progress_bar": tqdm_dict, "log": tqdm_dict, "generator_loss": generator_loss,}
        # )
        # return generator_output

    def _get_noise(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    def _get_fake_prediction(self, real: torch.Tensor) -> torch.Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_prediction = self.discriminator(fake)
        return fake_prediction

    def _get_discriminator_loss(self, real: torch.Tensor) -> torch.Tensor:
        real_prediction = self.discriminator(real)
        real_ground_truth = torch.ones_like(real_prediction)
        real_loss = self.criterion(real_prediction, real_ground_truth)

        fake_prediction = self._get_fake_prediction(real)
        fake_ground_truth = torch.zeros_like(fake_prediction)
        fake_loss = self.criterion(fake_prediction, fake_ground_truth)

        discriminator_loss = real_loss + fake_loss
        return discriminator_loss

    def _discriminator_step(self, real: torch.Tensor) -> torch.Tensor:
        discriminator_loss = self._get_discriminator_loss(real)
        self.log("loss/discriminator", discriminator_loss, on_epoch=True)
        return discriminator_loss
        # tqdm_dict = {"discriminator_loss": discriminator_loss}
        # discriminator_output = OrderedDict(
        #     {
        #         "loss": discriminator_loss,
        #         "progress_bar": tqdm_dict,
        #         "log": tqdm_dict,
        #         "discriminator_loss": discriminator_loss,
        #     }
        # )
        # return discriminator_output

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        result = None

        # Train Generator
        if optimizer_idx == 0:
            result = self._generator_step(real)
            self.generator_loss_collection.append(result.item())

        if optimizer_idx == 1:
            result = self._discriminator_step(real)
            self.discriminator_loss_collection.append(result.item())

        return result

    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", type=float, default=0.5)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--feature_maps_generator", type=int, default=64)
        parser.add_argument("--feature_maps_discriminator", type=int, default=64)
        parser.add_argument("--latent_dim", type=int, default=100)
        parser.add_argument("--learning_rate", type=float, default=0.002)
        parser.add_argument("--log_path", type=str, default=0.002)
        return parser
