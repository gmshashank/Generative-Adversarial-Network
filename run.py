import argparse
import torch
import torchvision
import pytorch_lightning as pl
from models import DCGAN
from models import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
import torchvision.transforms as transforms


def main(args=None):
    pl.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--data_dir", type=str, default="./", help="Path to data directory")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")

    script_args, _ = parser.parse_known_args(args)

    transform = transforms.Compose(
        [
            transforms.Resize((script_args.image_size, script_args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.570838093757629, 0.479552984237671, 0.491760671138763],
                std=[0.279659748077393, 0.309973508119583, 0.311098515987396],
            ),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=script_args.data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=script_args.batch_size, shuffle=True, num_workers=script_args.num_workers
    )
    image_channels = 3

    parser = DCGAN.add_to_argparse(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    args.weight_summary = "full"

    generator_loss_collection = []
    discriminator_loss_collection = []
    model = DCGAN(
        **vars(args),
        image_channels=image_channels,
        generator_loss_collection=generator_loss_collection,
        discriminator_loss_collection=discriminator_loss_collection,
    )

    logger = pl.loggers.TensorBoardLogger(save_dir="logs/")

    callbacks = [
        LatentDimInterpolator(interpolate_epoch_interval=5),
        TensorboardGenerativeModelImageSampler(num_samples=5),
    ]
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=args.num_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, dataloader)

    trainer.save_checkpoint(f"carGAN_custom_checkpoint_{args.num_epochs}.ckpt")

    # print(len(generator_loss_collection),len(discriminator_loss_collection))

    # print(f"generator_loss_collection: {generator_loss_collection}")
    with open("generator_loss_collection.txt", "w") as output:
        output.write(str(generator_loss_collection))

    # print(f"discriminator_loss_collection: {discriminator_loss_collection}")
    with open("discriminator_loss_collection.txt", "w") as output:
        output.write(str(discriminator_loss_collection))

    import matplotlib.pyplot as plt
    import numpy as np

    def smooth_curv(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return np.array(smoothed)

    plt.figure(figsize=(15, 8))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_loss_collection, "b-.", label="Generator", alpha=0.4)
    plt.plot(discriminator_loss_collection, "r-.", label="Discriminator", alpha=0.4)
    plt.plot(smooth_curv(generator_loss_collection, 0.95), "b")
    plt.plot(smooth_curv(discriminator_loss_collection, 0.95), "r")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("DCGAN_Loss.png")


if __name__ == "__main__":
    main()
