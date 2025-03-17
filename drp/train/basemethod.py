import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List, Dict, Any
from drp.handlers.logger import Logger
from drp.utils.iterative_mean import Mean
from drp.metrics.accuracy_at_k import TopKAccuracy
from drp.handlers.checkpoint import CheckPoint
from drp.utils.config import Config
from drp.utils.timer import Timer

from drp.metrics.r2_metric import DrpR2Metric
from drp.builder.VIT3d import EncoderViT
from drp.utils.backbone import generate_model, Simpler3DNet, Simpler3DNet_dilated, Custom_ConvLSTM


class BaseMethod:
    def __init__(self, config: Config, rank: int = 0, local_rank: int = 0, sampler_train: DistributedSampler = None):
        self._config = config

        self.rank = rank
        self.local_rank = local_rank
        self.sampler_train = sampler_train
        self.distributed = self.config["distributed"]
        self.gpu = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.local_rank if self.distributed else self.gpu

        self.backbone = self._create_model(num_classes=self.config["num_classes"])

        self.backbone = self._distribute_model(self.backbone)
        self.criterion = self._create_criterion(regression=self.config["regression"])

        if self.config["tqdm"]:
            self.train_loop = self._tqdm_loop(self.__inner_train_regression)
            self.valid_loop = self._tqdm_loop(self.__inner_validate_regression)
            self.test_loop = self._tqdm_loop(self.__inner_test_regression)
        else:
            self.train_loop = self._batch_loop(self.__inner_train_regression)
            self.valid_loop = self._batch_loop(self.__inner_validate_regression)
            self.test_loop = self._batch_loop(self.__inner_test_regression)

        if self.config["regression"]:

            if self.config["tqdm"]:
                self.train_loop = self._tqdm_loop(self.__inner_train_regression)
                self.valid_loop = self._tqdm_loop(self.__inner_validate_regression)
                self.test_loop = self._tqdm_loop(self.__inner_test_regression)
            else:

                self.train_loop = self._batch_loop(self.__inner_train_regression)
                self.valid_loop = self._batch_loop(self.__inner_validate_regression)
                self.test_loop = self._batch_loop(self.__inner_test_regression)

            self.r2_metric_train = DrpR2Metric().to(device=self.device)
            self.r2_metric_valid = DrpR2Metric().to(device=self.device)
            self.r2_metric_test = DrpR2Metric().to(device=self.device)

        else:

            if self.config["tqdm"]:
                self.train_loop = self._tqdm_loop(self.__inner_train_classification)
                self.valid_loop = self._tqdm_loop(self.__inner_validate_classification)

            else:
                self.train_loop = self._batch_loop(self.__inner_train_classification)
                self.valid_loop = self._batch_loop(self.__inner_validate_classification)

            self.TopKAccuracy_Train = TopKAccuracy(num_classes=
                                                   self.config["num_classes"],
                                                   top_k=(1, )).to(device=self.device)

            self.TopKAccuracy_Valid = TopKAccuracy(num_classes=
                                                   self.config["num_classes"],
                                                   top_k=(1, )).to(device=self.device)

        self.logger = None
        self.optimizer = None
        self.scheduler = None

    @property
    def config(self):
        return self._config.__dict__

    def _create_criterion(self, regression=True):
        if regression:
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()

    def _create_model(self, num_classes=1):
        if self.config["model"] == "resnet":

            model = generate_model(
                self.config["resnet"], n_classes=num_classes, activation=self.config["activation"]
            )
        elif self.config["model"] == "psimple":
            model = Simpler3DNet(activation=self.config["activation"])
        elif self.config["model"] == "psimple_dilated":
            model = Simpler3DNet_dilated(activation=self.config["activation"], dilation=self.config["dilation"])
        elif self.config["model"] == "lstm":
            model = Custom_ConvLSTM(hidden_dim=[8, 16, 32, 64])
        elif self.config["model"] == "transformer":
            model = EncoderViT(img_size=100, in_chans=1, embed_dim=256, depth=6, num_heads=8, num_classes=1,
                               use_seq_pool=True, mlp_ratio=4., norm_layer=nn.LayerNorm, patch_embed=False)
        else:
            raise EOFError

        return model

    def _distribute_model(self, model):

        if self.distributed:
            model = model.to(self.local_rank)
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = model.to(device=self.device)
            if self.device == "cuda" and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True
        return model

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        return [{"name": "backbone", "params": self.backbone.parameters()}]

    def _create_optimizer(self):
        optimizer_type = self.config['optimizer']
        try:
            optimizer = {
                "sgd": optim.SGD(
                    self.learnable_params,
                    lr=self.config["learning_rate"],
                    weight_decay=self.config["weight_decay"],
                    momentum=self.config["momentum"]
                ),
                "adam": optim.Adam(
                    self.learnable_params,
                    lr=self.config["learning_rate"],
                    weight_decay=self.config["weight_decay"],
                    betas=(0.8, 0.999)
                ),
                "adamw": optim.AdamW(
                    self.learnable_params,
                    lr=self.config["learning_rate"],
                    weight_decay=self.config["weight_decay"],
                    betas=(0.8, 0.999)

                )
            }[self.config['optimizer']]
        except KeyError:
            raise ValueError(f"Optimizer {optimizer_type} not in (sgd, adam)")
        return optimizer

    def _create_scheduler(self, optimizer):
        scheduler_type = self.config.get("scheduler", "none")

        if scheduler_type == "none":
            return None
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-4)
        elif scheduler_type == "step":
            return MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
        elif scheduler_type == "one_circle":
            return OneCycleLR(optimizer, max_lr=self.config["learning_rate"] * 10, steps_per_epoch=10,
                              epochs=self.config["epochs"])
        else:
            valid_schedulers = ['none', 'cosine', 'step', 'one_circle']
            raise ValueError(
                f"Scheduler '{scheduler_type}' not recognized. Valid options are: {', '.join(valid_schedulers)}")

    def _tqdm_loop(self, function):
        """
        This is a decorator that encapsulates the inner learning process.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times.
        """

        def new_function(epoch, loader, description, mean):
            if self.rank == 0:  # Initialize tqdm only on GPU 0
                timer = Timer()
                timer.start()
                size_by_batch = len(loader)
                step = max(size_by_batch // self.config["n_steps_by_batch"], 1)
                self.loss_tmp = []  # Initialize empty list to store loss values

                # with tqdm(loader, unit="batch", file=sys.stdout) as tepoch:
                #     for batch_idx, batch in enumerate(tepoch):
                #         tepoch.set_description(f"{description} {epoch}")
                #         loss = function(batch, epoch, batch_idx)  # Calculate loss for the batch
                #         timer.tick()
                #
                #         mean.update(loss)  # Update the mean with the new loss value
                #         tepoch.set_postfix(loss=mean.value)
                #
                #         if mean.iter % step == 0:
                #             # Append only the loss value to the list
                #             self.loss_tmp.append(mean.value)
                #
                #     # At the end of the epoch, calculate the average loss
                #     if self.loss_tmp:
                #         avg_loss = sum(self.loss_tmp) / len(self.loss_tmp)  # Calculate average loss
                #
                #         # Log the average loss to MLflow (or any logger you are using)
                #         self.logger.report_metric(
                #             metrics={f"{description}_avg_loss": avg_loss}, epoch=epoch)
                #
                #     self.loss_tmp.clear()  # Clear the list after logging
                for batch_idx, batch in enumerate(loader):
                    loss = function(batch, epoch, batch_idx)
                    timer.tick()
                    mean.update(loss)
                    if mean.iter % step == 0:
                        self.loss_tmp.append(mean.value)
                if self.loss_tmp:
                    avg_loss = sum(self.loss_tmp) / len(self.loss_tmp)  # Calculate average loss
                    self.logger.report_metric(
                                    metrics={f"{description}_avg_loss": avg_loss}, epoch=epoch)
                    self.loss_tmp.clear()  # Clear the list after logging

                timer.log()
                timer.stop()

            else:  # For other GPUs, run without tqdm progress bar
                for batch_idx, batch in enumerate(loader):
                    loss = function(batch, epoch, batch_idx)
                    mean.update(loss)

        return new_function

    def _batch_loop(self, function):
        def new_function(epoch, loader, description, mean):

            if self.rank == 0:  # Initialize tqdm only on GPU 0
                timer = Timer()
                timer.start()
                size_by_batch = len(loader)
                step = max(size_by_batch // self.config["n_steps_by_batch"], 1)
                self.loss_tmp = []
                for batch_idx, batch in enumerate(loader):
                    loss = function(batch, epoch, batch_idx)
                    timer.tick()

                    mean.update(loss)

                    if mean.iter % step == 0:
                        # Append the loss and iteration to the list
                        self.loss_tmp.append({"iteration": mean.iter, "loss": mean.value})

                # At the end of the epoch, log the accumulated losses
                for tmp in self.loss_tmp:
                    self.logger.report_metric(
                        metrics={f"{description}_losses": tmp["loss"]}, epoch=tmp["iteration"])
                self.loss_tmp.clear()
                timer.log()
                timer.stop()

            else:  # For other GPUs, run without tqdm progress bar

                for batch_idx, batch in enumerate(loader):
                    loss = function(batch, epoch, batch_idx)

                    mean.update(loss)

        return new_function

    def __inner_train_regression(self, batch, epoch, batch_idx):
        x, y_reg, _ = batch

        x, y = x.to(self.device), y_reg.to(self.device)

        self.optimizer.zero_grad()

        y_hat = self.backbone(x)

        loss = self.criterion(y_hat, y)
        loss.backward()
        params = [param["params"] for param in self.learnable_params]
        flat_params = [p for sublist in params for p in sublist]
        nn.utils.clip_grad_norm_(flat_params, 1.0)
        self.optimizer.step()

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)

            if self.rank == 0:
                self.r2_metric_train.update(y_hat, y)

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        else:
            self.r2_metric_train.update(y_hat, y)
        return loss.item()

    def __inner_validate_regression(self, batch, epoch, batch_idx):
        x, y_reg, _ = batch

        x, y = x.to(self.device), y_reg.to(self.device)

        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)

            if self.rank == 0:
                self.r2_metric_valid.update(y_hat, y)

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        else:
            self.r2_metric_valid.update(y_hat, y)

        return loss.item()

    def __inner_test_regression(self, batch, epoch, batch_idx):
        x, y_reg, _ = batch

        x, y = x.to(self.device), y_reg.to(self.device)

        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)

            if self.rank == 0:
                self.r2_metric_test.update(y_hat, y)

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        else:
            self.r2_metric_test.update(y_hat, y)

        return loss.item()

    def __inner_train_classification(self, batch, epoch, batch_idx):
        x, _, y_class = batch

        x, y = x.to(self.device), y_class.to(self.device)

        self.optimizer.zero_grad()

        y_hat = self.backbone(x)

        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)

            if self.rank == 0:
                self.TopKAccuracy_Train.update(y_hat, y)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        else:
            self.TopKAccuracy_Train.update(y_hat, torch.unsqueeze(y, dim=1))
        return loss.item()

    def __inner_validate_classification(self, batch, epoch, batch_idx):
        x, _, y_class = batch

        x, y = x.to(self.device), y_class.to(self.device)

        y_hat = self.backbone(x)

        loss = self.criterion(y_hat, y)

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)

            if self.rank == 0:
                self.TopKAccuracy_Valid.update(y_hat, y)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        else:
            self.TopKAccuracy_Valid.update(y_hat, torch.unsqueeze(y, dim=1))

        return loss.item()

    @torch.no_grad()
    def gather_across_gpus(self, data):
        gathered_data = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_data, data)
        gathered_data = torch.cat(gathered_data, dim=0)
        return gathered_data

    def train(self, epoch, train_loader, train_loss):
        self.backbone.train()
        return self.train_loop(epoch, train_loader, "Train", train_loss)

    def validate(self, epoch, valid_loader, valid_loss):
        self.backbone.eval()
        with torch.no_grad():
            return self.valid_loop(epoch, valid_loader, "Valid", valid_loss)

    def test(self, epoch, test_loader, test_loss):
        self.backbone.eval()
        with torch.no_grad():
            return self.test_loop(epoch, test_loader, "Test", test_loss)

    def _update_and_log_metrics(self, epoch, num_epochs, train_loss, valid_loss, test_loss=None, checkpoint=None):

        # computes metrics.
        if self.config["regression"]:
            metrics_train = self.r2_metric_train.compute(train_flag="Train")
            metrics_valid = self.r2_metric_valid.compute(train_flag="Validation")
            if self.config["regression"] is True:
                metrics_test = self.r2_metric_test.compute(train_flag="Test")

            precision = metrics_valid["r2_Validation"]
            self.r2_metric_train.reset()
            self.r2_metric_valid.reset()
            self.r2_metric_test.reset()
        else:
            metrics_train = self.TopKAccuracy_Train.compute(train=True)
            metrics_valid = self.TopKAccuracy_Valid.compute(train=False)
            print(metrics_train)
            print(metrics_valid)
            precision = metrics_valid["top_1_accuracy_Validation"]
            self.TopKAccuracy_Train.reset()
            self.TopKAccuracy_Valid.reset()

        if self.config["scheduler"] != "none":
            metrics_valid["lr"] = self.scheduler.get_last_lr()[0]
        else:
            metrics_valid["lr"] = self.config["learning_rate"]
        self.logger.report_metric(metrics_train, epoch)
        self.logger.report_metric(metrics_valid, epoch)
        if self.config["regression"] is True:
            self.logger.report_metric(metrics_test, epoch)

            checkpoint.update(
                run_id=self.logger.run_id,
                epoch=epoch,
                model=self.backbone.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                train_loss=train_loss.state_dict(),
                valid_loss=valid_loss.state_dict(),
                test_loss=test_loss.state_dict(),
                precision=precision
            )
        else:
            checkpoint.update(
                run_id=self.logger.run_id,
                epoch=epoch,
                model=self.backbone.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                train_loss=train_loss.state_dict(),
                valid_loss=valid_loss.state_dict(),
                precision=precision
            )

        self.logger.log_checkpoint(checkpoint, monitor="precision")
        self.logger.report_metric(metrics={"epoch": epoch}, epoch=epoch)

    def fit(self, train_loader, valid_loader, test_loader=None, run_id=None):
        num_epochs = self.config["epochs"]
        train_loss = Mean.create("ewm")
        valid_loss = Mean.create("ewm")
        if self.config["regression"] is True:
            test_loss = Mean.create("ewm")

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

        checkpoint = CheckPoint(run_id)
        checkpoint.init_model(self.backbone)
        checkpoint.init_optimizer(self.optimizer)
        checkpoint.init_scheduler(self.scheduler)
        if self.config["regression"] is True:
            checkpoint.init_loss(train_loss, valid_loss, test_loss)
        else:
            checkpoint.init_loss(train_loss, valid_loss)
        if self.rank == 0:
            self.logger = Logger(self._config, run_id=run_id)
            self.logger.start()
            self.logger.set_signature(train_loader)
            self.logger.summary()
        else:
            self.logger = None

        for epoch in range(checkpoint.epoch + 1, num_epochs):
            if self.distributed:
                self.sampler_train.set_epoch(epoch)

            if self.rank == 0:
                print(f"Epoch {epoch}")  # Print epoch number only on the primary GPU

            # Execute training and validation for the epoch
            self.train(epoch, train_loader, train_loss)
            if self.distributed:
                dist.barrier()
            self.validate(epoch, valid_loader, valid_loss)

            if self.rank == 0:
                if self.config["scheduler"] != "none":
                    self.scheduler.step()
                # Update and log metrics
                if self.config["regression"] is True:
                    self._update_and_log_metrics(epoch, num_epochs, train_loss, valid_loss, test_loss, checkpoint)
                else:
                    self._update_and_log_metrics(epoch=epoch,
                                                 num_epochs=num_epochs,
                                                 train_loss=train_loss,
                                                 valid_loss=valid_loss,
                                                 checkpoint=checkpoint)

        if self.rank == 0:
            self.logger.close()
        if self.distributed:
            torch.distributed.destroy_process_group()  # clean up

