from drp.train.basemethod import BaseMethod
from torch import nn
from drp.utils.config import Config
from torch.utils.data import DistributedSampler
from drp.handlers.checkpoint import CheckPoint


class FineTune(BaseMethod):
    def __init__(self,
                 config: Config,
                 rank: int = 0,
                 local_rank: int = 0,
                 sampler_train: DistributedSampler = None,
                 ) -> None:
        super().__init__(config, rank, local_rank, sampler_train)
        dim = self.backbone.fc.weight.shape[1]

        if self.config["SSL"] == "SimCLR" or self.config["SSL"] == "MixSim":

            self.backbone.fc = nn.Sequential(
                nn.Linear(dim, 2048, bias=False),
                nn.ReLU(),
                nn.Linear(2048, 128, bias=False)
            )
        elif self.config["SSL"] == "MoCo":

            self.backbone.fc = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, 256)
            )
        elif self.config["SSL"] == "BYOL" or self.config["SSL"] == "SimBYL":

            self.backbone.fc = nn.Sequential(
                nn.Linear(dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 256)
            )
        elif self.config["SSL"] == "SimSiam":
            proj_hidden_dim = 512
            proj_output_dim = 128
            self.backbone.fc = nn.Sequential(nn.Linear(dim, proj_hidden_dim, bias=False),
                                             nn.BatchNorm1d(proj_hidden_dim),
                                             nn.ReLU(inplace=True),  # first layer
                                             nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
                                             nn.BatchNorm1d(proj_hidden_dim),
                                             nn.ReLU(inplace=True),  # second layer
                                             nn.Linear(proj_hidden_dim, proj_output_dim),
                                             nn.BatchNorm1d(proj_output_dim, affine=False))  # output layer

        checkpoint_ = CheckPoint(run_id=self.config["ckp_runid"], best_ckp=False, finetune_ssl=True)
        checkpoint_.init_model(self.backbone)

        self.backbone.fc = nn.Linear(512, self.config["num_classes"])
        self.backbone = self._distribute_model(self.backbone)

        if not self.config["finetune"]:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
