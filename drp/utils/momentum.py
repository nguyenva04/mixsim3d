import math

import torch
from torch import nn
from threading import Lock

@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    """Copies the parameters of the online network to the momentum network.

    Args:
        online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
        momentum_net (nn.Module): momentum network (e.g. momentum backbone,
            momentum projection, etc...).
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        """Updates momentum parameters using exponential moving average.

        Args:
            base_tau (float, optional): base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.996.
            final_tau (float, optional): final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 1.0.
        """

        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )


class Queue:
    def __init__(self, queue_size, proj_output_dim, device) -> None:
        self.queue_size = queue_size
        self.proj_output_dim = proj_output_dim
        self.device = device

        # create the queue
        self.queue = torch.randn(self.proj_output_dim, self.queue_size)
        self.queue = nn.functional.normalize(self.queue, dim=0).to(self.device)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.lock = Lock()

    def state_dict(self):
        return {"queue": self.queue, "queue_ptr": self.queue_ptr}

    def load_state_dict(self, state_dict):
        self.queue = state_dict["queue"]
        self.queue_ptr = state_dict["queue_ptr"]

    @property
    def value(self):
        with self.lock:
            queue = self.queue
        return queue

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


