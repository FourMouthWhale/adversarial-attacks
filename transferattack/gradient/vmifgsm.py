"""
VMI-FGSM: Variable - Momentum Iterative Fast Gradient Sign Method
"""
import torch

from ..utils import *
from ..attack import Attack

class VMIFGSM(Attack):
    def __init__(self, model_name, attack='VMI-FGSM', epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1., targeted=False, 
                 random_start=False, norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor

    def get_variance(self, data, delta, label, cur_grad, momentum, **kwargs):
        grad = 0
        for _ in range(self.num_neighbor):
            logits = self.get_logits(self.transform(data+delta+torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device), momentum=momentum))
            loss = self.get_loss(logits, label)
            grad += self.get_grad(loss, delta)
        return grad / self.num_neighbor - cur_grad
    
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        momentum, variance = 0, 0
        for _ in range(self.epoch):
            logits = self.get_logits(self.tansform(data+delta, momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad+variance, momentum)
            variance = self.get_variance(data, delta, label, grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()