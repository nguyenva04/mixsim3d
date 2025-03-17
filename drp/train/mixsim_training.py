import torch
import math
import numpy as np

import torch.distributed as dist
import torch.nn.functional as F

from torch import nn
from typing import List, Dict, Any

from drp.utils.timer import Timer
from drp.builder.simclr import SimclrEncoder

from drp.utils.config import Config
from drp.handlers.logger import Logger

from drp.utils.iterative_mean import Mean
from drp.handlers.checkpoint import CheckPoint

from drp.train.basemethod import BaseMethod
from torch.utils.data.distributed import DistributedSampler


class LamdaUpdater:
    def __init__(self, base_lamda: float = 0.01, final_lamda: float = 0.99):
        """Updates lamda parameters using cosine annealing.

        Args:
            base_lamda (float, optional): Base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.01.
            final_lamda (float, optional): Final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.99.
        """

        super().__init__()

        assert 0 <= base_lamda <= 1
        assert 0 <= final_lamda <= 1 and base_lamda <= final_lamda

        self.base_lamda = base_lamda
        self.cur_lamda = base_lamda
        self.final_lamda = final_lamda

    def update_lamda(self, current_epoch: int, total_epochs: int):
        """Computes the next value for the lamda using cosine annealing.

        Args:
            current_epoch (int): The current epoch.
            total_epochs (int): The total number of epochs in the training.
        """

        self.cur_lamda = (
            self.final_lamda
            - (self.final_lamda - self.base_lamda) * (math.cos(math.pi * current_epoch / total_epochs) + 1) / 2
        )


class MixSim3d(BaseMethod):
    def __init__(self,
                 config: Config,
                 rank: int = 0,
                 local_rank: int = 0,
                 sampler_train: DistributedSampler = None,
                 ) -> None:
        super().__init__(config, rank, local_rank, sampler_train)

        model = self._create_model()
        self.model = SimclrEncoder(model)
        self.model = self._distribute_model(self.model)

        self.std = 0.0
        self.batch_size = self.config["batch_size"]

        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.cuda.empty_cache()
            self.batch_size //= self.config["world_size"]

        self.lamda_updater = LamdaUpdater()

        if self.config["tqdm"]:
            self.train_loop = self._tqdm_loop(self.__inner_train)
        else:
            self.train_loop = self._batch_loop(self.__inner_train)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        return [{"name": "model", "params": self.model.parameters()}]

    def _tqdm_loop(self, function):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times
        """
        def new_function(epoch, loader, description, mean):
            if self.rank == 0:  # Initialize tqdm only on GPU 0
                timer = Timer()
                timer.start()
                size_by_batch = len(loader)
                step = max(size_by_batch // self.config["n_steps_by_batch"], 1)
                self.loss_tmp = []
                self.std_tmp = []

                for batch_idx, batch in enumerate(loader):
                    loss = function(batch, epoch, batch_idx)
                    timer.tick()
                    mean.update(loss)
                    if mean.iter % step == 0:
                        self.loss_tmp.append(mean.value)
                        self.std_tmp.append(np.float64(self.std))
                # At the end of the epoch, log the accumulated losses
                if self.loss_tmp:
                    avg_loss = sum(self.loss_tmp) / len(self.loss_tmp)  # Calculate average loss
                if self.std_tmp:
                    avg_std = sum(self.std_tmp) / len(self.std_tmp)  # Calculate average loss

                # Log the average loss to MLflow (or any logger you are using)
                self.logger.report_metric(
                    metrics={f"{description}_loss": avg_loss}, epoch=epoch)
                self.logger.report_metric(
                    metrics={f"{description}_std": avg_std}, epoch=epoch)

                self.loss_tmp.clear()
                self.std_tmp.clear()
                timer.log()
                timer.stop()

            else:  # For other GPUs, run without tqdm progress bar

                for batch_idx, batch in enumerate(loader):
                    loss = function(batch, epoch, batch_idx)

                    mean.update(loss)

        return new_function

    def loss_fn(self, out_1, out_2):
        out_1 = nn.functional.normalize(out_1, dim=-1)
        out_2 = nn.functional.normalize(out_2, dim=-1)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.config["temperature"])
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.config["temperature"])
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def __inner_train(self, batch, epoch, batch_idx):
        (x1, x2), _, y = batch

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y = y.to(self.device)
        x_mixed = (1 - self.lamda_updater.cur_lamda)*x1 + self.lamda_updater.cur_lamda*x2

        self.optimizer.zero_grad()

        z1, z_mixed, z2 = self.model(x1), self.model(x_mixed.to(self.device)), self.model(x2)

        loss = (self.loss_fn(z1, z_mixed) + self.loss_fn(z1, z2))/2
        loss.backward()
        self.optimizer.step()

        if self.distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()

            z1 = self.gather_across_gpus(z1)
            z_mixed = self.gather_across_gpus(z_mixed)

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z_mixed_std = F.normalize(z_mixed, dim=-1).std(dim=0).mean()
        self.std = (z1_std + z_mixed_std) / 2

        return loss.item()

    def _update_and_log_metrics(self, epoch, num_epochs, train_loss, valid_loss=None, checkpoint=None):

        checkpoint.update(
            run_id=self.logger.run_id,
            epoch=epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            train_loss=train_loss.state_dict(),
        )

        self.logger.report_metric(metrics={"lamda": self.lamda_updater.cur_lamda}, epoch=epoch)
        self.logger.report_metric({"lr": self.scheduler.get_last_lr()[0]}, epoch)
        self.logger.report_metric(metrics={"epoch": epoch}, epoch=epoch)
        self.logger.log_checkpoint(checkpoint)

    def fit(self, train_loader, valid_loader=None, run_id=None):
        num_epochs = self.config["epochs"]
        train_loss = Mean.create("ewm")

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

        checkpoint = CheckPoint(run_id)
        checkpoint.init_model(self.model)
        checkpoint.init_optimizer(self.optimizer)
        checkpoint.init_scheduler(self.scheduler)
        checkpoint.init_loss(train_loss)

        if self.rank == 0:
            self.logger = Logger(self._config, run_id=run_id)
            self.logger.start()
            self.logger.set_signature(train_loader)
            self.logger.summary()
        else:
            self.logger = None

        for epoch in range(checkpoint.epoch + 1, num_epochs):
            if self.rank == 0:
                self.lamda_updater.update_lamda(epoch, num_epochs)

            if self.distributed:
                self.sampler_train.set_epoch(epoch)

            if self.rank == 0:
                print(f"Epoch {epoch}")  # Print epoch number only on the primary GPU

            # Execute training and validation for the epoch
            self.train(epoch, train_loader, train_loss)
            if self.distributed:
                dist.barrier()

            if self.rank == 0:
                self.scheduler.step()
                # Update and log metrics
                self._update_and_log_metrics(epoch=epoch, num_epochs=num_epochs, train_loss=train_loss, checkpoint=checkpoint)

        if self.rank == 0:
            self.logger.close()
        if self.distributed:
            torch.distributed.destroy_process_group()  # clean up

