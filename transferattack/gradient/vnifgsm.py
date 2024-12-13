"""
VNI-FGSM: Variable - Nesterov Iterative Fast Gradient Sign Method
"""
import torch

from ..utils import * 
from .vmifgsm import VMIFGSM

class VNIFGSM(VMIFGSM):
    def __init__(self, model_name, attack='VNI-FGSM', epsilon=16 / 255, alpha=1.6 / 255, beta=1.5, num_neighbor=20, epoch=10, decay=1, targeted=False, 
                 random_start=False, norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, alpha, beta, num_neighbor, epoch, decay, targeted, random_start, norm, loss, device)

    def transform(self, x, momentum):
        return x + self.alpha * self.decay * momentum