import os
import uuid
import torch
import mlflow
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from typing import Optional

from threading import Lock

from drp.utils.time_utils import Timer
from drp.utils.config import Config
from drp.handlers.checkpoint import CheckPoint
from drp.utils.display import lrfind_plot


class Logger:
    @staticmethod
    def _run_name(root_name):
        return f"{root_name}_{str(uuid.uuid1())[:8]}"

    @property
    def config(self):
        return self._config.__dict__

    @property
    def experiment_id(self):
        experiment = mlflow.get_experiment_by_name(self.config["project"])
        if experiment is None:
            return mlflow.create_experiment(self.config["project"])
        else:
            return experiment.experiment_id

    def __init__(self, config: Config, run_id: str = None) -> None:
        self._config = config
        self.run_id = run_id
        self.agg = matplotlib.rcParams["backend"]
        self.signature = None
        self.lock = Lock()
        mlflow.set_tracking_uri(self.config["tracking_uri"])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _params(self):
        # take care, some config parameters are saved by mlflow.
        # When you run it again, these parameters can not change between two runs.
        params = self._config.__dict__
        del params["run_id"]
        del params["epochs"]
        return params

    def start(self):
        matplotlib.use("agg")
        active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            run_name=self._run_name(self.config["task"]),
        )
        self.run_id = active_run.info.run_id
        mlflow.log_params(self._params())

    def close(self):
        matplotlib.use(self.agg)
        mlflow.end_run()

    def __str__(self):
        msg = f"{self.config['task']}: {self.config['resnet']} (seed: {self.config['seed']})\n"
        # msg += f"Device : {self.device}\n"
        msg += f"matplotlib backend: {matplotlib.rcParams['backend']}, interactive: {matplotlib.is_interactive()}\n"
        active_run = mlflow.active_run()
        if active_run:
            msg += f"Name: {active_run.info.run_name}\n"
            msg += f"Experiment_id: {active_run.info.experiment_id}\n"
            msg += f"Run_id: {self.run_id}\n"
        if self.signature is not None:
            msg += f"Signature: {self.signature}"

        return msg

    def set_signature(self, loader):
        with Timer() as timer:
            X, y, idx = next(iter(loader))
            if len(X) == 2:
                self.signature = f"X : {X[0].shape}, Y : {y.shape} idx: {idx} time: {timer}s size: {len(loader.dataset)}"
            else:
                self.signature = f"X : {X.shape}, Y : {y.shape} idx: {idx} time: {timer}s size: {len(loader.dataset)}"

    def summary(self):
        print(str(self))
        mlflow.log_text(str(self), "summary.txt")

    def log_checkpoint(self, checkpoint: CheckPoint, monitor=None):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckp_name = os.path.join(tmpdirname, f"checkpoint.pt")
            torch.save(checkpoint.params, ckp_name)
            mlflow.log_artifact(local_path=ckp_name, artifact_path="checkpoints")
            if monitor is not None:
                if checkpoint.best_precision is None:
                    checkpoint.update(best_precision=checkpoint.precision)
                if checkpoint.best_precision <= checkpoint.precision:
                    ckp_name = os.path.join(tmpdirname, f"best_checkpoint.pt")
                    torch.save(checkpoint.params, ckp_name)
                    mlflow.log_artifact(local_path=ckp_name, artifact_path="checkpoints")
                    checkpoint.update(best_precision=checkpoint.precision)

    def save_model(self):
        model = self.checkpoint.backbone
        signature = self.checkpoint.signature
        mlflow.pytorch.log_model(model, "backbone", signature=signature)

    def report_metric(self, metrics: dict[str, float], epoch: Optional[int] = None):
        with self.lock:
            mlflow.log_metrics(metrics, step=epoch)

    def report_figure(self, figure, description):
        with self.lock:
            mlflow.log_figure(figure=figure, artifact_file=description)
        plt.close(figure)

    def report_findlr(self, lrs, losses):
        fig = lrfind_plot(lrs, losses)
        with self.lock:
            mlflow.log_figure(figure=fig, artifact_file="find_lr.jpg")
        plt.close(fig)
