# -*- coding: utf-8 -*-
"""
Trainer class for training the MR-VAE model.
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
from torch.optim.lr_scheduler import StepLR

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, Timer
from ignite.metrics import RunningAverage, Loss, MeanSquaredError

from torchvision.utils import make_grid

from src.trainer.utils import gpu_config
from src.config.logger_initialization import setup_custom_logger
from src.trainer.utils import randomness_seed
from src.config.paths_configuration import ROOT_DIR

logger = setup_custom_logger(__name__)
img_dir = os.path.join(ROOT_DIR, 'reports/figures/reconstruction/',
                       datetime.datetime.now().strftime("%Y_%m_%d--%Hh%Mmn%S"))


def kld_loss(mu, log_var):
    """
    Calculates the Kullback-Leibler divergence loss.
    Args:
        mu (torch.Tensor): Mean of the latent distribution.
        log_var (torch.Tensor): Log variance of the latent distribution.

    Returns:
        torch.Tensor: The Kullback-Leibler divergence loss.

    Notes:
        see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch_sum(1 + log_var - mu.pow(2) - log_var.exp())


def trainer(model, train_loader, test_loader, args):
    """
    Trainer function.
    """
    # arguments parameters:
    nb_classes = 10
    labels_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    epochs = args.epochs
    lr = args.lr
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
    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    bce_loss = nn.BCELoss(reduction='sum')
    # set up neptune:
    if neptune_log:
        run = neptune.init(project=args.neptune_project,
                           api_token=args.neptune_api_token,
                           tags=['MR-VAE', 'mnist', 'test'],
                           source_files=[__file__])
        run['parameters'] = args.__dict__

    def train_batch(engine, batch):
        model.train()
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        x_recons, _, mu, log_var = model(x)
        BCE = bce_loss(x_recons, x)
        KLD = kld_loss(mu, log_var)
        loss = BCE + beta * KLD
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'recon_loss': BCE.item(), 'kl_loss': KLD.item()}

    def val_batch(engine, batch):
        model.eval()
        x, _ = batch
        x = x.to(device)
        with no_grad():
            x_recons, _, mu, log_var = model(x)
            return x_recons, x, mu, log_var

    # Setup trainer and evaluator
    train_engine = Engine(train_batch)
    val_engine = Engine(val_batch)

    timer = Timer(average=True)
    # attach running average metrics
    monitoring_metrics = ["RunningAverage_loss",
                          "RunningAverage_recon_loss",
                          "RunningAverage_kl_loss"]
    RunningAverage(alpha=alpha, output_transform=lambda x: x["loss"]).attach(train_engine,
                                                                             "RunningAverage_loss")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["recon_loss"]).attach(train_engine,
                                                                      "RunningAverage_recon_loss")
    RunningAverage(alpha=alpha,
                   output_transform=lambda x: x["kl_loss"]).attach(train_engine,
                                                                   "RunningAverage_kl_loss")
    # attach metrics to evaluator

    MeanSquaredError(output_transform=lambda x: [x[0], x[1]]).attach(val_engine, 'Eval_MSE')
    Loss(bce_loss, output_transform=lambda x: [x[0], x[1]]).attach(val_engine, 'Eval_BCE')
    Loss(kld_loss, output_transform=lambda x: [x[2], x[3]]).attach(val_engine, 'Eval_KLD')

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
        pbar.log_message(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
        timer.reset()

    if crash_iteration > 0:
        @train_engine.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):  # sourcery no-metrics skip: raise-specific-error
            raise Exception(f"STOP at {engine.state.iteration}")

    def print_logs(engine, dataloader, mode):
        val_engine.run(dataloader, max_epochs=1)
        metrics = val_engine.state.metrics
        avg_mse = metrics['Eval_MSE']
        avg_bce = metrics['Eval_BCE']
        avg_kld = metrics['Eval_KLD']
        avg_loss = avg_bce + avg_kld * beta
        print(
            mode + "Results - Epoch {} - Avg loss: {:.2f} Avg mse: {:.2f} Avg bce: {:.2f} Avg "
                   "kld: {:.2f} "
            .format(engine.state.epoch, avg_loss, avg_mse, avg_bce, avg_kld))
        if neptune_log:
            run[f'{mode}/train_loss'].log(avg_loss)
            run[f'{mode}/train_MSE'].log(avg_mse)
            run[f'{mode}/train_BCE'].log(avg_bce)
            run[f'{mode}/train_KLD'].log(avg_kld)

    train_engine.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training')
    train_engine.add_event_handler(Events.EPOCH_COMPLETED, print_logs, test_loader, 'Validation')

    @train_engine.on(Events.EPOCH_COMPLETED)
    def print_trainer_logs(engine):
        """
        Compute metrics.
        """
        lr_scheduler.step()  # update learning rate
        avg_loss = engine.state.metrics['RunningAverage_loss']
        avg_bce = engine.state.metrics['RunningAverage_recon_loss']
        avg_kld = engine.state.metrics['RunningAverage_kl_loss']
        print("Trainer Results - Epoch {} - Lr:{} - Avg loss: {:.2f} Avg bce: {:.2f} Avg "
              "kld: {:.2f}".format(engine.state.epoch, lr_scheduler, avg_loss, avg_bce, avg_kld))

    def compare_images(engine, save_img=False):
        model.eval()
        test_dataset = test_loader.dataset
        epoch = engine.state.epoch
        n = 10
        fig = plt.figure(figsize=(16, 4.5))
        targets = test_dataset.targets.numpy()
        t_idx = {_class: np.where(targets == _class)[0][0] for _class in range(nb_classes)}

        for i, _class in enumerate(range(nb_classes)):
            ax = plt.subplot(2, n, i + 1)
            img = test_dataset[t_idx[_class]][0].unsqueeze(0).to(device)
            with no_grad():
                rec_img, _, _, _ = model(img)
            plt.imshow(img.cpu().squeeze().numpy(), cmap=plt.cm.Greys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(f'T{i}: {labels_map[_class]}')
            if i == n // 2:
                ax.set_title('Original images')

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap=plt.cm.Greys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
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
        """Saves a grid of generated digits"""
        epoch = engine.state.epoch
        lista_images = []
        z = randn(n_row * n_row, args.latent_dim)
        z = z.to(device)
        gen_img = model.decode(z).cpu()
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
        l_latent_space, l_y = [], []
        for img, y in dataloader:
            img = img.to(device)
            img = img.view(img.shape[0], -1)
            mu, log_var = model.encode(img)
            latent_space = model.reparameterize(mu, log_var)
            latent_space = latent_space.cpu().detach().numpy()
            l_latent_space.extend(latent_space)
            l_y.extend(y)
        return np.asarray(l_latent_space), l_y

    def show_latent_dataloader(labels):
        latent_space, y = calculate_latent(test_loader)
        pca = PCA(n_components=2)
        components = pca.fit_transform(latent_space)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plt.scatter(x=components[:, 0], y=components[:, 1], c=y, cmap='tab10')
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
    def handle_exception(engine, e):
        if not isinstance(e, KeyboardInterrupt) or engine.state.iteration <= 1:
            raise e
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
