from typing import List, Optional, Tuple
import numpy as np
import torch
import pytorch_lightning as pl
import torchvision


class LatentDimInterpolator(pl.Callback):
    # Interpolates the latent space for a model by setting all dims to zero and stepping through first two dims increasing one unit at a time
    def __init__(
        self,
        interpolate_epoch_interval: int = 20,
        range_start: int = -5,
        range_end: int = 5,
        steps: int = 11,
        num_samples: int = 2,
        normalize: bool = True,
    ):
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.steps = steps

    def interpolate_latent_space(self, pl_module: pl.LightningModule, latent_dim: int) -> List[torch.Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    z = torch.zeros(self.num_samples, latent_dim, device=pl_module.device)

                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    img = pl_module(z)
                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)
        pl_module.train()
        return images

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(pl_module, latent_dim=pl_module.hparams.latent_dim)
            images = torch.cat(images, dim=0)

            num_rows = self.steps
            grid = torchvision.utils.make_grid(images, nrow=num_rows, normalize=self.normalize)
            gird_title = f"{pl_module.__class__.__name__}_latent_space"
            trainer.logger.experiment.add_image(gird_title, grid, global_step=trainer.global_step)


class TensorboardGenerativeModelImageSampler(pl.Callback):
    # Generate images and logs to tensorboard.

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z)
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        grid_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(grid_title, grid, global_step=trainer.global_step)
