import argparse
import torch
import torchvision
import pytorch_lightning as pl
from models import DCGAN
import torchvision.transforms as transforms


def main(args=None):
    pl.seed_everything(9)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--data_dir", type=str, default="./", help="Path to data directory")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")

    script_args, _ = parser.parse_known_args(args)

    transform = transforms.Compose(
        [
            transforms.Resize(script_args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.570838093757629, 0.479552984237671, 0.491760671138763],
                std=[0.279659748077393, 0.309973508119583, 0.311098515987396],
            ),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=script_args.data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=script_args.batch_size, shuffle=True, num_workers=2)
    image_channels = 3

    parser = DCGAN.add_to_argparse(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    model = DCGAN(**vars(args), image_channels=image_channels)

    callbacks = []
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=args.num_epochs, callbacks=callbacks)
    trainer.fit(model, dataloader)

    trainer.save_checkpoint(f"carGAN_custom_checkpoint_{args.num_epochs}.ckpt")


if __name__ == "__main__":
    main()
