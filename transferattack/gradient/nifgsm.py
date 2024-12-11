from ..utils import *
from .mifgsm import MIFGSM

class NIFGSM(MIFGSM):
    def __init__(self, model_name, attack='NI-FGSM', epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, 
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device)

    def transform(self, x, momentum):
        return x + self.alpha * self.decay * momentum