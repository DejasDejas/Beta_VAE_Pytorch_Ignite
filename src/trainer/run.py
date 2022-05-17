# -*- coding: utf-8 -*-
"""
Run the training process with the specified configuration passed as command line arguments.
"""
import argparse

from src.trainer.utils import str2bool
from src.trainer.trainer import trainer
from src.data.make_mnist_dataset import load_mnist_data
from src.models.model import VAE
from src.config.logger_initialization import setup_custom_logger

log = setup_custom_logger(__name__)


def main(args):
    """
    Run the training process with the specified configuration passed as command line arguments.
    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # data:
    train_loader, test_loader = load_mnist_data(batch_size_train=args.batch_size)
    img_shape = train_loader.dataset[0][0].shape
    # model:
    model = VAE(args.latent_dim, img_shape)
    # log.info(f"Model: {model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Number of trainable parameters: {num_params:,}")
    # train:
    trainer(model, train_loader, test_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MR-VAE model.")
    # randomness seed:
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Training settings:
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1000,
        help="input batch size for validation (default: 1000)"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=20,
        help="dimensionality of the latent space (default: 20)"
    )
    parser.add_argument(
        "--beta", type=float, default=1,
        help="beta for the KL divergence (default: 1)"
    )
    # validation batch size
    # optimiser
    # loss function
    parser.add_argument(
        "--log_interval", type=int, default=100,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--model_dir", type=str, default="models/", help="model directory for save models trained",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=5, help="Checkpoint training every X epochs",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to the checkpoint .pt file to resume training from ",
    )
    parser.add_argument(
        "--crash_iteration", type=int, default=-1, help="Iteration at which to raise an exception",
    )
    parser.add_argument(
        "--early_stopping", default=False, type=str2bool,
        help="True if you want to use early stopping.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.98,
        help="smoothing constant for exponential moving averages"
    )

    # Model settings:
    # number of hidden layers'
    parser.add_argument(
        "--dropout", default=False, type=str2bool,
        help="True if you want to use dropout.",
    )
    # network weight initialisation
    # activation function

    # Neptune parameters:
    parser.add_argument(
        "--neptune_project", type=str, default="MR-VAE", help="Neptune project name"
    )
    parser.add_argument(
        "--neptune_log", default=True, type=str2bool,
        help="True if you want to use early neptune logs.",
    )
    parser.add_argument(
        "--pbar", default=True, type=str2bool,
        help="True if you want to use progress bar.",
    )
    parser.add_argument(
        "--neptune_api_token",
        type=str,
        default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9h"
                "cHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNTYwZjdjZS05M2FkLTQ1M2MtOTgxZi0xOWNhZjU2MmRl"
                "YWYifQ==",
        help="Neptune API token.",
    )

    arguments = parser.parse_args()
    main(arguments)
