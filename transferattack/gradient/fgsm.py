from ..utils import *
from ..attack import Attack

class FGSM(Attack):
    def __init__(self, model_name, attack='FGSM', epsilon=16/255, targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon
        self.epoch = 1
        self.decay = 0