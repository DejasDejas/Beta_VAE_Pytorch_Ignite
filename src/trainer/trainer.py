# -*- coding: utf-8 -*-
# pylint: disable = logging-fstring-interpolation, import-error, too-many-locals, invalid-name
# pylint: disable = unused-argument, broad-except, logging-format-interpolation, unused-variable
# pylint: disable = too-many-statements, no-name-in-module, consider-using-f-string
# pylint: disable = logging-not-lazy
"""
Trainer class for training the model, evaluating the model,saving and monitoring the model.
"""
import os
import warnings
import traceback
import datetime
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import neptune.new as neptune
from torch import no_grad, nn, randn
from torch import load as load_torch
from torch import sum as torch_sum
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, Timer, EarlyStopping
from ignite.metrics import RunningAverage

from torchvision.utils import make_grid

from src.trainer.utils import gpu_config
from src.config.logger_initialization import setup_custom_logger
from src.trainer.utils import randomness_seed
from src.config.paths_configuration import ROOT_DIR

logger = setup_custom_logger(__name__)
img_dir = os.path.join(ROOT_DIR, 'reports/figures/reconstruction/',
                       datetime.datetime.now().strftime("%Y_%m_%d--%Hh%Mmn%S"))


def kld_loss(_mu, log_var):
    """
    Calculates the Kullback-Leibler divergence loss.
    Args:
        _mu (torch.Tensor): Mean of the latent distribution.
        log_var (torch.Tensor): Log variance of the latent distribution.

    Returns:
        torch.Tensor: The Kullback-Leibler divergence loss.

    Notes:
        see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch_sum(1 + log_var - _mu.pow(2) - log_var.exp())


def trainer(model, train_loader, test_loader, args):
    """
    Trains the model.
    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        args (argparse.Namespace): The arguments.

    Returns:
        None
    """
    # arguments parameters:
    nb_classes = 10
    labels_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    epochs = args.epochs
    learning_rate = args.lr
    checkpoint_every = args.checkpoint_every
    crash_iteration = args.crash_iteration
    alpha = args.alpha
    model_dir = os.path.join(ROOT_DIR, args.model_dir)
    resume_from = args.resume_from
    beta = args.beta
    use_pbar = args.pbar
    neptune_log = args.neptune_log

    # model and training parameters config:
    model, device = gpu_config(model)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
    bce_loss = nn.BCELoss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='sum')

    # set up neptune:
    if neptune_log:
        run = neptune.init(project=args.neptune_project,
                           api_token=args.neptune_api_token,
                           tags=['MR-VAE', 'mnist', 'test'],
                           source_files=[__file__])
        run['parameters'] = args.__dict__

    def train_batch(engine, batch):
        """
        Inner training loop.
        Args:
            engine (ignite.engine.Engine): The training engine.
            batch (tuple): The batch of data.

        Returns:
            dict: The loss.
        """
        model.train()
        img, _ = batch
        img = img.to(device)
        optimizer.zero_grad()
        img_recons, _, _mu, log_var = model(img)
        BCE = bce_loss(img_recons, img)
        KLD = kld_loss(_mu, log_var)
        loss = BCE + beta * KLD
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'BCE': BCE.item(), 'KLD': KLD.item()}

    def val_batch(engine, batch):
        """
        Inner validation loop.
        Args:
            engine (ignite.engine.Engine): The validation engine.
            batch (tuple): The batch of data.

        Returns:
            dict: The loss.
        """
        model.eval()
        img, _ = batch
        img = img.to(device)
        with no_grad():
            img_recons, _, _mu, log_var = model(img)
            BCE = bce_loss(img_recons, img)
            KLD = kld_loss(_mu, log_var)
            loss = BCE + beta * KLD
            MSE = mse_loss(img_recons, img)
            return {'loss': loss.item(), 'BCE': BCE.item(), 'KLD': KLD.item(), 'MSE': MSE.item()}

    # Setup trainer and evaluator
    train_engine = Engine(train_batch)
    val_engine = Engine(val_batch)

    # attach early stopping:
    def score_function(engine):
        """
        Inner score function.
        Args:
            engine (ignite.engine.Engine): The training engine.
        Returns:
            float: The score.
        """
        val_loss = engine.state.metrics['Eval_loss']
        return -val_loss

    handler_early_stopping = EarlyStopping(patience=10,
                                           score_function=score_function,
                                           trainer=train_engine)
    val_engine.add_event_handler(Events.COMPLETED, handler_early_stopping)

    timer = Timer(average=True)
    # attach running average metrics
    monitoring_metrics = ["RunningAverage_loss",
                          "RunningAverage_recon_loss",
                          "RunningAverage_kl_loss"]
    RunningAverage(alpha=alpha, output_transform=lambda x: x["loss"]).attach(train_engine,
                                                                             "RunningAverage_loss")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["BCE"]).attach(train_engine,
                                                               "RunningAverage_recon_loss")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["KLD"]).attach(train_engine,
                                                               "RunningAverage_kl_loss")
    # attach metrics to evaluator
    RunningAverage(alpha=alpha, output_transform=lambda x: x["loss"]).attach(val_engine,
                                                                             "Eval_loss")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["MSE"]).attach(val_engine, "Eval_MSE")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["BCE"]).attach(val_engine, "Eval_BCE")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["KLD"]).attach(val_engine, "Eval_KLD")

    # attach progress bar
    pbar = ProgressBar(persist=True)
    if use_pbar:
        pbar.attach(train_engine, metric_names=monitoring_metrics)

    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(
        train_engine,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    # adding handlers using `trainer.on` decorator API
    @train_engine.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        """
        Inner function for print the time per epoch.
        Args:
            engine (ignite.engine.Engine): The training engine.
        Returns:
            None
        """
        pbar.log_message(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
        timer.reset()

    if crash_iteration > 0:
        @train_engine.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):  # sourcery no-metrics skip: raise-specific-error
            raise Exception(f"STOP at {engine.state.iteration}")

    def print_logs(engine, dataloader, mode):
        """
        Inner function for print the logs.
        Args:
            engine (ignite.engine.Engine): The training engine.
            dataloader (ignite.contrib.handlers.tqdm_logger.TqdmLogger): The dataloader.
            mode (str): The mode of the dataloader.

        Returns:
            None
        """
        val_engine.run(dataloader, max_epochs=1)
        metrics = val_engine.state.metrics
        avg_loss = metrics['Eval_loss']
        avg_mse = metrics['Eval_MSE']
        avg_bce = metrics['Eval_BCE']
        avg_kld = metrics['Eval_KLD']
        logger.info(
            mode + "Results - Epoch {} - Avg loss: {:.2f} Avg mse: {:.2f} Avg bce: {:.2f} Avg "
                   "kld: {:.2f}"
            .format(engine.state.epoch, avg_loss, avg_mse, avg_bce, avg_kld))
        if neptune_log:
            run[f'{mode}/avg_loss'].log(avg_loss)
            run[f'{mode}/MSE'].log(avg_mse)
            run[f'{mode}/BCE'].log(avg_bce)
            run[f'{mode}/KLD'].log(avg_kld)
            if mode == 'Training':
                run['lr'].log(optimizer.param_groups[0]['lr'])

    train_engine.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training')
    train_engine.add_event_handler(Events.EPOCH_COMPLETED, print_logs, test_loader, 'Validation')

    @train_engine.on(Events.EPOCH_COMPLETED)
    def print_trainer_logs(engine):
        """
        Inner function for print the logs during training.
        Args:
            engine (ignite.engine.Engine): The training engine.

        Returns:
            None
        """
        avg_loss = engine.state.metrics['RunningAverage_loss']
        avg_bce = engine.state.metrics['RunningAverage_recon_loss']
        avg_kld = engine.state.metrics['RunningAverage_kl_loss']
        lr_scheduler.step(avg_loss)  # update learning rate
        logger.info("Trainer Results - Epoch {} - Lr:{} - Avg loss: {:.2f} Avg bce: {:.2f} Avg "
                    "kld: {:.2f}".format(engine.state.epoch, optimizer.param_groups[0]["lr"],
                                         avg_loss, avg_bce, avg_kld))

    def compare_images(engine, save_img=False):
        """
        Inner function for compare the images reconstructed by the model with the original images.
        Args:
            engine (ignite.engine.Engine): The training engine.
            save_img (bool): Whether to save the images.

        Returns:
            None
        """
        model.eval()
        test_dataset = test_loader.dataset
        epoch = engine.state.epoch
        n_img = 10
        fig = plt.figure(figsize=(16, 4.5))
        targets = test_dataset.targets.numpy()
        t_idx = {_class: np.where(targets == _class)[0][0] for _class in range(nb_classes)}

        for i, _class in enumerate(range(nb_classes)):
            ax = plt.subplot(2, n_img, i + 1)
            img = test_dataset[t_idx[_class]][0].unsqueeze(0).to(device)
            with no_grad():
                rec_img, _, _, _ = model(img)
            plt.imshow(img.cpu().squeeze().numpy(), cmap=plt.cm.Greys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(f'T{i}: {labels_map[_class]}')
            if i == n_img // 2:
                ax.set_title('Original images')

            ax = plt.subplot(2, n_img, i + 1 + n_img)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap=plt.cm.Greys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_img // 2:
                ax.set_title('Reconstructed images')
        if save_img:
            plt.savefig(f'{img_dir}reconstructed_epoch_{str(epoch)}.png')
        plt.show()
        # neptune images:
        if neptune_log:
            run[f"reconstruction/comparison_image_{epoch}"].upload(
                neptune.types.File.as_image(fig))
        plt.close(fig)

    def sample_image(engine, n_row):
        """
        Inner function for sample the images reconstructed by the model.
        Args:
            engine (ignite.engine.Engine): The training engine.
            n_row (int): The number of rows in the image.

        Returns:
            None
        """
        epoch = engine.state.epoch
        lista_images = []
        hidden = randn(n_row * n_row, args.latent_dim)
        hidden = hidden.to(device)
        gen_img = model.decode(hidden).cpu()
        gen_img = gen_img.view(n_row * n_row, 1, 28, 28)
        lista_images.extend(gen_img)
        img = make_grid(lista_images, n_row)
        figure, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img.permute(1, 2, 0))
        plt.show()
        # neptune images:
        if neptune_log:
            run[f"generated_images/sample_img_{epoch}"].upload(
                neptune.types.File.as_image(figure))
        plt.close(figure)

    def calculate_latent(dataloader):
        """
        Inner function for calculate the latent space.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader.

        Returns:
            latent_space (list): The latent space.
            labels list: The labels of the images.
        """
        l_latent_space, l_y = [], []
        for img, label in dataloader:
            img = img.to(device)
            img = img.view(img.shape[0], -1)
            _mu, log_var = model.encode(img)
            latent_space = model.reparameterize(_mu, log_var)
            latent_space = latent_space.cpu().detach().numpy()
            l_latent_space.extend(latent_space)
            l_y.extend(label)
        return np.asarray(l_latent_space), l_y

    def show_latent_dataloader(labels):
        """
        Inner function for show the latent space.
        Args:
            labels (list): The labels of the images.

        Returns:
            None
        """
        latent_space, label = calculate_latent(test_loader)
        pca = PCA(n_components=2)
        components = pca.fit_transform(latent_space)
        fig = plt.subplots(1, 1, figsize=(7, 7))
        plt.scatter(x=components[:, 0], y=components[:, 1], c=label, cmap='tab10')
        cbar = plt.colorbar()
        cbar.set_ticks(list(range(10)))
        cbar.set_ticklabels(labels)
        # neptune images:
        if neptune_log:
            run["evaluation/latent_space_loader"].upload(
                neptune.types.File.as_image(fig))
        plt.close(fig)

    train_engine.add_event_handler(Events.STARTED, compare_images, save_img=False)
    train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), compare_images, save_img=True)
    train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), sample_image, n_row=10)
    train_engine.add_event_handler(Events.COMPLETED(every=1), show_latent_dataloader,
                                   labels=labels_map)

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": train_engine,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(model_dir, require_empty=False),
        n_saved=None,
        global_step_transform=lambda *_: train_engine.state.epoch,
    )
    train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_every),
                                   training_checkpoint)
    if resume_from is not None:
        pbar.log_message(f"Resume from the checkpoint: {resume_from}")
        checkpoint = load_torch(os.path.join(ROOT_DIR, resume_from))
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    # adding handlers using `trainer.on` decorator API
    @train_engine.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, exception):
        """
        Inner function for handling the exception.
        Args:
            engine (Engine): The engine.
            exception (Exception): The exception.

        Returns:
            None
        """
        if not isinstance(exception, KeyboardInterrupt) or engine.state.iteration <= 1:
            raise exception
        engine.terminate()
        warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")

    # Run the training engine
    try:
        # Synchronize random states
        randomness_seed(0)
        train_engine.run(train_loader, max_epochs=epochs)
    except Exception as _e:
        pbar.log_message(f"Exception: {_e}")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        logger.info(template.format(type(_e).__name__, _e.args))
        logger.info(traceback.format_exc(), _e)

    pbar.log_message("Training finished.")
    pbar.close()
    if neptune_log:
        run.stop()
