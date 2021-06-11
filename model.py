import numpy as np
import torch
import torch.nn as nn


class SpikieFC(nn.Module):
    def __init__(self, N_in, N_out, initW=None):
        super().__init__()
        if initW is not None:
            self.W = initW
        else:
            self.W = nn.Parameter(torch.Tensor(
                0.1 * np.random.rand(N_out, N_in)
            ))

    def forward(self, x):
        return torch.matmul(x, self.W)
