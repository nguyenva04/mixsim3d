import os
import torch

from drp.data.dataset import Drp3dMMapDatasetSSL
from drp.train.simsiam_training import SimSiamTraining
from drp.data.data_transform import PairTransform
from drp.utils.config import Config

from drp.utils.arguments import parse_args
from torch.utils.data import DataLoader
from drp.utils.hostlist import expand_hostlist
from drp.utils.ddp_setup import setup, prepare_dataset


def main(
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 0,
        config: Config = None
):
    pair_transform = PairTransform(config)
    train_dataset = Drp3dMMapDatasetSSL(config, train_flag="Train", transform=pair_transform, finetune=True)

    if config.distributed:
        hostnames = expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        setup(rank, world_size, IP_ad=hostnames[0])  # initialize ddp
        train_dataloader, _, _, train_sampler = prepare_dataset(trainset=train_dataset,
                                                                config=config,
                                                                rank=rank,
                                                                world_size=world_size)
        trainer = SimSiamTraining(
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

        trainer = SimSiamTraining(config=config)

    trainer.fit(
        train_loader=train_dataloader, run_id=config.run_id)


if __name__ == "__main__":

    args = parse_args()
    config = Config.create_from_args(args)
    torch.manual_seed(config.seed)
    torch.rand(config.seed).numpy()
    # get SLURM variables
    if config.distributed:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        config.world_size = int(os.environ['SLURM_NTASKS'])
        main(rank=rank, local_rank=local_rank, world_size=config.world_size, config=config)
    else:
        main(config=config)
