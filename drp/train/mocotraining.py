import torch
import sys
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import math
from drp.utils.config import Config
import torch.distributed as dist
from drp.handlers.logger import Logger
from typing import List, Dict, Any
from drp.train.basemethod import BaseMethod
from drp.utils.timer import Timer
from drp.builder.moco import MoCo, QueueModule

from drp.utils.iterative_mean import Mean
from drp.handlers.checkpoint import CheckPoint
from torch.utils.data.distributed import DistributedSampler


class MoCoV2(BaseMethod):
    def __init__(self,
                 config: Config,
                 rank: int = 0,
                 local_rank: int = 0,
                 sampler_train: DistributedSampler = None,
                 ) -> None:
        super().__init__(config, rank, local_rank, sampler_train)

        self.local_rank = local_rank
        backbone = self._create_model()

        base_encoder_momentum = self._create_model()
        backbone.fc = nn.Linear(backbone.fc.weight.shape[1], self.config["proj_output_dim"])
        base_encoder_momentum.fc = nn.Linear(base_encoder_momentum.fc.weight.shape[1], self.config["proj_output_dim"])

        self.model = MoCo(base_encoder=backbone,
                          base_encoder_momentum=base_encoder_momentum,
                          m=self.config["momentum_base_tau"],
                          T=self.config["temperature"],
                          mlp=True,
                          distributed=self.distributed)

        self.queue = QueueModule(dim=self.config["proj_output_dim"], K=self.config["queue_size"])
        self.queue = self.queue.to(self.device)
        self.model = self._distribute_model(self.model)

        self.std = 0.0
        self.batch_size = self.config["batch_size"]

        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.cuda.empty_cache()
            self.batch_size //= self.config["world_size"]

        if self.config["tqdm"]:
            self.train_loop = self._tqdm_loop(self.__inner_train_momentum)
        else:
            self.train_loop = self._batch_loop(self.__inner_train_momentum)


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
                with tqdm(loader, unit="batch", file=sys.stdout) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        tepoch.set_description(f"{description} {epoch}")
                        loss = function(batch, epoch, batch_idx)
                        timer.tick()

                        mean.update(loss)

                        tepoch.set_postfix(loss=mean.value)
                        if mean.iter % step == 0:
                            # Append the loss and iteration to the list
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

    def loss_function(self, q, k):

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.queue.detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.config["temperature"]

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels, reduction="mean")
        return loss

    def __inner_train_momentum(self, batch, epoch, batch_idx):

        (x1, x2), _, y = batch

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        self.optimizer.zero_grad()

        q1, k2 = self.model(x1, x2)
        q2, k1 = self.model(x2, x1)

        loss = (self.loss_function(q1, k2)
                + self.loss_function(q2, k1)) / 2

        loss.backward()
        self.optimizer.step()

        if self.distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()

            q1 = self.gather_across_gpus(q1)
            q2 = self.gather_across_gpus(q2)
            k1 = self.gather_across_gpus(k1)
            k2 = self.gather_across_gpus(k2)

        self.model.module.momentum_update_key_encoder()

        # ------- update queue -------
        keys = torch.cat([k1, k2], dim=0)

        self.queue.dequeue_and_enqueue(keys)

        # calculate std of features
        q1_std = F.normalize(q1, dim=-1).std(dim=0).mean()
        q2_std = F.normalize(q2, dim=-1).std(dim=0).mean()
        k1_std = F.normalize(k1, dim=-1).std(dim=0).mean()
        k2_std = F.normalize(k2, dim=-1).std(dim=0).mean()
        self.std = (q1_std + q2_std + k1_std + k2_std) / 4
        return loss.item()

    def _update_and_log_metrics(self, epoch, num_epochs, train_loss, valid_loss=None, checkpoint=None):

        checkpoint.update(
            run_id=self.logger.run_id,
            epoch=epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            train_loss=train_loss.state_dict(),
            queue=self.queue.state_dict()
        )

        if self.config["mixed"]:
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
        checkpoint.init_queue(self.queue)
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
            if self.rank == 0 and self.config["mixed"] is True:
                self.lamda_updater.update_lamda(epoch, num_epochs, fixed_lamda=self.config["fixed_lamda"])
            if self.distributed:
                self.sampler_train.set_epoch(epoch)

            if self.rank == 0:
                print(f"Epoch {epoch}")  # Print epoch number only on the primary GPU

            # Execute training and validation for the epoch
            self.train(epoch, train_loader, train_loss)

            if self.rank == 0:
                self.scheduler.step()
                # Update and log metrics
                self._update_and_log_metrics(epoch=epoch, num_epochs=num_epochs, train_loss=train_loss, checkpoint=checkpoint)

        if self.rank == 0:
            self.logger.close()
        if self.distributed:
            torch.distributed.destroy_process_group()  # clean up


