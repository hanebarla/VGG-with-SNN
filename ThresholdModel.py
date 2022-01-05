import torch
from statistics import mean
import math


class ExponentialModel():
    def __init__(self, alpha=1.0, beta=1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def __call__(self, time, Vth):
        y = 1 - math.exp(-1 * self.alpha * time)
        return self.beta * y

class AdaptiveModel():
    def __init__(self, chagestep) -> None:
        self.acc_history = []
        self.firerate_history = []
        self.overfire_history = []
        self.entropy_history = []
        self.state = 0
        """
        State
        ---------
        0: start
        1: Vthを下げる必要があるとき
        2: Vthを上げる必要があるとき
        """
        self.changestep = chagestep

    def __call__(self, time, Vth):
        """
        if self.state == 0:
            self.state = 1
            return Vth

        starttime = time - self.changestep
        overfire = mean(self.overfire_history[starttime:time])
        if overfire > 0:
            self.state = 2
            return Vth + 0.05

        self.state = 1
        return Vth - 0.05
        """
        return Vth + (self.entropy_history[-1] / 10)

    def firelog(self, fire):
        self.firerate_history.append(fire[0])
        self.overfire_history.append(fire[1])

    def entropylog(self, entropy):
        self.entropy_history.append(entropy)
