import os
import torch.distributed as dist
from typing import Tuple, Optional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from drp.utils.config import Config


def setup(rank, world_size, IP_ad):
    os.environ['MASTER_ADDR'] = IP_ad
    os.environ['MASTER_PORT'] = '12355'
    init_str = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'

    dist.init_process_group("nccl", init_method=init_str, rank=rank, world_size=world_size)


def prepare_dataset(
    trainset: Dataset = None,
    validset: Optional[Dataset] = None,
    testset: Optional[Dataset] = None,
    config: Optional[Config] = None,
    rank=0,
    world_size=1
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], DistributedSampler]:

    # Create the distributed sampler for the train dataset
    sampler_train = DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    # Create the DataLoader for the train dataset
    trainloader = DataLoader(
        trainset,
        batch_size=int(config.batch_size / world_size),
        shuffle=False,  # Shuffle is handled by the sampler
        sampler=sampler_train,
        num_workers=config.num_workers,
        drop_last=True
    )

    if validset is not None:
        sampler_valid = DistributedSampler(
            validset, shuffle=False, num_replicas=world_size, rank=rank
        )
        validoader = DataLoader(
            validset,
            batch_size=int(config.batch_size / world_size),
            shuffle=False,  # Shuffle is handled by the sampler
            sampler=sampler_valid,
            num_workers=config.num_workers,
            drop_last=True
        )
    else:
        validoader = None

    if testset is not None:
        sampler_test = DistributedSampler(
            testset, shuffle=False, num_replicas=world_size, rank=rank
        )
        testloader = DataLoader(
            testset,
            batch_size=int(config.batch_size / world_size),
            shuffle=False,
            sampler=sampler_test,
            num_workers=config.num_workers,
            drop_last=False  # Drop last might not be needed for testing
        )
    else:
        testloader = None

    # Return all DataLoaders and the training sampler
    return trainloader, validoader, testloader, sampler_train