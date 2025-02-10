from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class Nadam(Optimizer):
    """
    Custom optimizer class.
    """

    momentum: float
    state: dict[Any, dict[str, Tensor]]
    learning_rate: float
    parameters: Any

    def __init__(self, params, defaults: dict[str, Any], lr=1e-3, momentum=0.9):
        """
        :param params: parameters of the model.
        :param defaults:
        :param lr: learning rate of the optimizer.
        :param momentum:
        """
        super().__init__(params, defaults)
        self.learning_rate = lr
        self.momentum = momentum
        self.state = dict()
        self.parameters = params
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))

                # Step Method

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))
                mom = self.state[p]['mom']
                mom = self.momentum * mom - group['lr'] * p.grad.data
                p.data += mom

    def zero_grad(self, set_to_none: bool = True) -> None:
        try:
            super().zero_grad(set_to_none)
            print('Gradients are zero now.')
        except Exception as e:
            print(e.__cause__)
            print('Error occurred in Zero grad method in Nadam.')
