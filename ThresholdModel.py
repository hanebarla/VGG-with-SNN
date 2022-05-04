import torch
from statistics import mean
import math

#  Adaptive のモデルと同じように作成
class ExponentialModel():
    def __init__(self, alpha=1.0, beta=1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, time, Vth):
        y = 1 - math.exp(-1 * self.alpha * time)
        return self.beta * y

class ReverseExponentialModel():
    def __init__(self, alpha=1.0, beta=1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, time, Vth):
        y = math.exp(-1 * self.alpha * time)
        return self.beta * y

class OverExponentialModel():
    def __init__(self, alpha=1.0, beta=9.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, time, Vth):
        y = math.exp(-1 * self.alpha * time)
        return (self.beta * y) + 1.0

class ConstExponentialModel():
    def __init__(self, alpha, timestep) -> None:
        self.alpha = alpha
        self.beta = 1 - ((math.exp(-alpha*timestep) + 1) / (alpha * timestep))

    def __call__(self, time, Vth):
        y = math.exp(-1 * self.alpha * time)
        return y + self.beta

class CosineExponentialModel():
    def __init__(self, alpha, beta, timestep, offset=1.0, amplitude=0.5):
        self.alpha = alpha
        self.freq = 2 * math.pi * beta / timestep # frequency of cosine
        self.offset = offset
        self.amplitude = amplitude

    def __call__(self, time, Vth):
        y = self.amplitude * math.cos(self.freq * time) * math.exp(-1 * self.alpha * time)
        return y + self.offset


class AdaptiveLIFModel():
    def __init__(self, tau, alpha) -> None:
        self.tau = tau
        self.alpha = alpha
        self.vthIni = 1.0
        self.vthTheta = 0.0
        self.watchVarStanderd = 0.5

    def __call__(self, watchVar):
        d_vthTheta = (-self.vthTheta + (self.alpha * (watchVar - self.watchVarStanderd))) / self.tau
        self.vthTheta += d_vthTheta

        return self.vthIni + self.vthTheta
