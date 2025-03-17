import os
import torch
import mlflow
import tempfile
from typing import Optional


class CheckPoint:
    def __init__(self, run_id=None, best_ckp=False, finetune_ssl=False) -> None:
        self.params = {"run_id": run_id}
        if run_id is not None:
            run = mlflow.get_run(self.run_id)
            assert run is not None, f"unable to find run {self.run_id}"
            with tempfile.TemporaryDirectory() as tmpdir:
                local_artifact = mlflow.artifacts.download_artifacts(
                    run.info.artifact_uri, dst_path=tmpdir
                )
                if best_ckp:
                    ck_path = os.path.join(local_artifact, "checkpoints", "best_checkpoint.pt")
                else:
                    ck_path = os.path.join(local_artifact, "checkpoints", "checkpoint.pt")

                self.params = torch.load(ck_path)
                if finetune_ssl:

                    self.params["model"] = {k: v for k, v in self.params["model"].items()
                                            if k.startswith("module.encoder_q")}
                    new_model_params = {}
                    for k in list(self.params["model"].keys()):
                        new_key = k.replace("module.encoder_q.", "")
                        new_model_params[new_key] = self.params["model"][k]
                    self.params["model"] = new_model_params
                else:
                    new_model_params = {}
                    for k in list(self.params["model"].keys()):
                        new_key = k.replace("module.", "")
                        new_model_params[new_key] = self.params["model"][k]
                    self.params["model"] = new_model_params

        else:
            self.params["epoch"] = -1
            self.params["best_precision"] = None

    @property
    def run_id(self):
        return self.params["run_id"]

    @property
    def epoch(self):
        return self.params["epoch"]

    @property
    def precision(self):
        return self.params["precision"]

    @property
    def queue(self):
        return self.params["queue"]

    @property
    def best_precision(self):

        return self.params["best_precision"]

    def init_model(self, model):
        if self.run_id is not None:
            model.load_state_dict(self.params["model"])

    def init_queue(self, queue):
        if self.run_id is not None:
            queue.load_state_dict(self.params["queue"])

    def init_momentum_model(self, momentum_model):
        if self.run_id is not None:
            momentum_model.load_state_dict(self.params["momentum_model"])

    def init_predictor(self, model):
        if self.run_id is not None:
            model.load_state_dict(self.params["predictor"])

    def init_fc(self, fc):
        if self.run_id is not None:
            fc.load_state_dict(self.params["fc"])

    def init_cur_tau(self, momentum_update):
        if self.run_id is not None:
            momentum_update.cur_tau = self.params["cur_tau"]

    def init_optimizer(self, optimizer):
        if self.run_id is not None:
            optimizer.load_state_dict(self.params["optimizer"])

    def init_scheduler(self, scheduler):
        if self.run_id is not None:
            scheduler.load_state_dict(self.params["scheduler"])

    def init_loss(self, train_loss, valid_loss: Optional[float] = None, test_loss: Optional[float] = None):
        if self.run_id is not None:
            train_loss.load_state_dict(self.params["train_loss"])
            if valid_loss is not None:
                valid_loss.load_state_dict(self.params["valid_loss"])
            if test_loss is not None:
                test_loss.load_state_dict(self.params["test_loss"])

    def update(self, **kwargs):
        self.params.update(**kwargs)

