import torch
from torch.optim import Adam, AdamW
import torch.nn as nn
from enum import StrEnum
import numpy as np
import algos.sac.core as core


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.obs_buf =  np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.bufs = [self.obs_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf]

    def store(self, obs, act, rew, next_obs, done):
        entries = [obs, next_obs, act, rew, done]
        for buf, entry in zip(self.bufs, entries):
            buf[self.ptr] = entry.detach().cpu().numpy() if isinstance(entry, torch.Tensor) else entry
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = torch.randint(0, self.size, size=(batch_size,))
        batch = dict(obs=self.obs_buf[idxs], 
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}


class ActivationType(StrEnum):
    RELU = "relu"
    TANH = "tanh"
    GELU = "gelu"


class OptimizerType(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"


ACTIVATIONS = {
    ActivationType.RELU: nn.ReLU,
    ActivationType.TANH: nn.Tanh,
    ActivationType.GELU: nn.GELU,
}

OPTIMIZERS = {
    OptimizerType.ADAM: Adam,
    OptimizerType.ADAMW: AdamW
}