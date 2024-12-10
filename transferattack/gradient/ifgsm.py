from ..utils import *
from ..attack import Attack

class IFGSM(Attack):
    def __init__(self, model_name, attack='I-FGSM', epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False,
                  norm='infty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 0