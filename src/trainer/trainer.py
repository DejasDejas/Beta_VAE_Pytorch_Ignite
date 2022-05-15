# -*- coding: utf-8 -*-
"""
Trainer class for training the MR-VAE model.
"""
import os
import warnings
import traceback
from matplotlib import pyplot as plt

import neptune.new as neptune
import torch
from torch.nn import BCELoss
from torch import no_grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, Timer
from ignite.metrics import RunningAverage, Loss, MeanSquaredError

from torchvision.utils import save_image, make_grid

from src.trainer.utils import gpu_config
from src.config.logger_initialization import setup_custom_logger
from src.trainer.utils import randomness_seed
from src.config.paths_configuration import ROOT_DIR

logger = setup_custom_logger(__name__)
img_dir = os.path.join(ROOT_DIR, 'reports/figures/reconstruction/')


def kld_loss(x_pred, x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def trainer(model, train_loader, test_loader, args):
    """
    Trainer function.
    """
    # arguments parameters:
    epochs = args.epochs
    lr = args.lr
    checkpoint_every = args.checkpoint_every
    crash_iteration = args.crash_iteration
    alpha = args.alpha
    model_dir = os.path.join(ROOT_DIR, args.model_dir)
    resume_from = args.resume_from

    # model and training parameters config:
    model, device = gpu_config(model)
    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    bce_loss = BCELoss(reduction='sum')
    # set up neptune:
    run = neptune.init(project=args.neptune_project,
                       api_token=args.neptune_api_token,
                       tags=['MR-VAE', 'mnist', 'test'],
                       source_files=[__file__])
    run['parameters'] = args.__dict__

    # batch of training data:
    for batch in train_loader:
        x, y = batch
        break
    fixed_images = x.to(device)

    def train_batch(engine, batch):
        model.train()
        x, _ = batch
        x = x.to(device)
        x = x.view(-1, 784)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        BCE = bce_loss(recon_x, x)
        KLD = kld_loss(recon_x, x, mu, logvar)
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'recon_loss': BCE.item(), 'kl_loss': KLD.item()}

    def val_batch(engine, batch):
        model.eval()
        x, _ = batch
        x = x.to(device)
        x = x.view(-1, 784)
        with no_grad():
            recon_x, mu, logvar = model(x)
            kwargs = {'mu': mu, 'logvar': logvar}
            return recon_x, x, kwargs

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
    Loss(kld_loss).attach(val_engine, 'Eval_KLD')

    # attach progress bar
    pbar = ProgressBar(persist=True)
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
        def _(engine):  # sourcery skip: raise-specific-error
            raise Exception(f"STOP at {engine.state.iteration}")

    def print_logs(engine, dataloader, mode):
        val_engine.run(dataloader, max_epochs=1)
        metrics = val_engine.state.metrics
        avg_mse = metrics['Eval_MSE']
        avg_bce = metrics['Eval_BCE']
        avg_kld = metrics['Eval_KLD']
        avg_loss = avg_bce + avg_kld
        print(
            mode + "Results - Epoch {} - Avg mse: {:.2f} Avg loss: {:.2f} Avg bce: {:.2f} Avg "
                   "kld: {:.2f} "
            .format(engine.state.epoch, avg_mse, avg_loss, avg_bce, avg_kld))
        if mode == 'Training':
            run['train_loss'].log(avg_loss)
            run['train_recon_loss'].log(avg_bce)
            run['train_kl_loss'].log(avg_kld)
        elif mode == 'Validation':
            run['val_loss'].log(avg_loss)
            run['val_recon_loss'].log(avg_bce)
            run['val_kl_loss'].log(avg_kld)

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
        print("Trainer Results - Epoch {} - Avg loss: {:.2f} Avg bce: {:.2f} Avg kld: {:.2f}"
              .format(engine.state.epoch, avg_loss, avg_bce, avg_kld))

    def compare_images(engine, save_img=False):
        epoch = engine.state.epoch
        reconstructed_images = model(fixed_images.view(-1, 784))[0].view(-1, 1, 28, 28)
        comparison = torch.cat([fixed_images, reconstructed_images])
        if save_img:
            save_image(comparison.detach().cpu(), f'{img_dir}reconstructed_epoch_{str(epoch)}.png',
                       nrow=8)
        comparison_image = make_grid(comparison.detach().cpu(), nrow=8)
        fig = plt.figure(figsize=(5, 5))
        output = plt.imshow(comparison_image.permute(1, 2, 0))
        plt.title(f'Epoch {str(epoch)}')
        plt.show()
        # neptune images:
        run["comparison_image"].upload(fig)

    train_engine.add_event_handler(Events.STARTED, compare_images, save_img=False)
    train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), compare_images, save_img=True)

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
        checkpoint = torch.load(os.path.join(ROOT_DIR, resume_from))
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
    run.stop()
