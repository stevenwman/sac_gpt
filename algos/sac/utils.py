import torch
from torch.optim import Adam, AdamW
import torch.nn as nn
from enum import StrEnum
import numpy as np
import algos.sac.core as core
from torch import Tensor

class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.obs_buf =  torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device='cpu')
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device='cpu')
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32, device='cpu')
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device='cpu')
        self.done_buf = torch.zeros(size, dtype=torch.float32, device='cpu')
        self.ptr, self.size, self.max_size = 0, 0, size

        self.bufs = [self.obs_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf]

    # def store(self, obs: Tensor, act: Tensor, rew: Tensor, next_obs: Tensor, done: Tensor):
    #     entries = [obs, next_obs, act, rew, done]
    #     for buf, entry in zip(self.bufs, entries):
    #         buf[self.ptr] = entry.detach().cpu()
    #     self.ptr = (self.ptr+1) % self.max_size
    #     self.size = min(self.size+1, self.max_size)

    def store_batch(self, obs: Tensor, act: Tensor, rew: Tensor, next_obs: Tensor, done: Tensor):
        # Assumes the first dimension of all tensors is num_envs
        num_envs = obs.shape[0]
        entries = [obs, next_obs, act, rew, done]
        # Calculate the indices to store the batch
        start_idx = self.ptr
        end_idx = start_idx + num_envs
        
        # Handle buffer wrap-around
        if end_idx > self.max_size:
            num_part1 = self.max_size - start_idx
            num_part2 = end_idx - self.max_size
            for buf, entry in zip(self.bufs, entries):
                buf[start_idx:] = entry[:num_part1].cpu()
                buf[:num_part2] = entry[num_part1:].cpu()
        else:
            for buf, entry in zip(self.bufs, entries):
                buf[start_idx:end_idx] = entry.cpu()

        # Update pointer and size
        self.ptr = (self.ptr + num_envs) % self.max_size
        self.size = min(self.size + num_envs, self.max_size)

    def sample_batch(self, batch_size):
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