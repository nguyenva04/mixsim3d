import os
import torch
from torch.utils.data import DataLoader

from drp.train.finetune import FineTune
from drp.train.finetune_mae import FineTuneMAE
from drp.data.dataset import Drp3dMMapDataset
from drp.utils.config import Config

from drp.utils.arguments import parse_args
from drp.utils.hostlist import expand_hostlist
from drp.utils.ddp_setup import setup, prepare_dataset


def main(
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 0,
        config: Config = None
):

    train_dataset = Drp3dMMapDataset(config, train_flag="Train", type_model="idx")
    valid_dataset = Drp3dMMapDataset(config, train_flag="Valid", type_model="idx")

    if config.distributed:
        hostnames = expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        setup(rank, world_size, IP_ad=hostnames[0])  # initialize ddp
        train_dataloader, valid_dataloader, test_dataloader, train_sampler = prepare_dataset(
            trainset=train_dataset, validset=valid_dataset, config=config, rank=rank, world_size=world_size
        )
        if config.SSL == "MAE":
            trainer = FineTuneMAE(config=config,
                                  rank=rank,
                                  local_rank=local_rank,
                                  sampler_train=train_sampler)
        else:
            trainer = FineTune(
                config=config,
                rank=rank,
                local_rank=local_rank,
                sampler_train=train_sampler,
            )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=True,
        )

        if config.SSL == "MAE":
            trainer = FineTuneMAE(config=config)
        else:
            trainer = FineTune(config=config)

    test_dataloader = None
    trainer.fit(
        train_loader=train_dataloader, valid_loader=valid_dataloader, test_loader=test_dataloader, run_id=config.run_id
    )


if __name__ == "__main__":

    args = parse_args()
    config = Config.create_from_args(args)
    torch.manual_seed(config.seed)
    torch.rand(config.seed).numpy()
    # get SLURM variables
    if config.distributed:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        config.gpus = int(os.environ['SLURM_NTASKS'])
        main(rank=rank, local_rank=local_rank, world_size=world_size, config=config)
    else:
        main(config=config)





