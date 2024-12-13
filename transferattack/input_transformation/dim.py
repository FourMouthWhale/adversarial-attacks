import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class DIM(MIFGSM):
    def __init__(self, model_name, attack='DIM', epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, resize_rate=1.1, diversity_prob=0.5, 
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device)
        if resize_rate < 1:
            raise Exception("Error! The resize rate shoule be larger than 1!")
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def tansform(self, x, **kwargs):
        if torch.rand(1) > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), type=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return F.interpolate(padded, size=[img_size, img_size], mode="bilinear", align_corners=False)