import torch

from ..utils import *
from .mifgsm import MIFGSM

class EMIFGSM(MIFGSM):
    def __init__(self, model_name, attack='EMI-FGSM', epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, num_sample=11, radius=7, sample_method='linear', 
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device)
        self.num_sample = num_sample
        self.radius = radius
        self.sample_method = sample_method.lower()

    def transform(self, x, grad, **kwargs):
        factors = np.linspace(-self.radius, self.radius, num=self.num_sample)
        return torch.concat([x+factor*self.alpha*grad for factor in factors])
    
    def get_loss(self, logits, label):
        return -self.loss(logits, label.repeat(self.num_sample)) if self.targeted else self.loss(logits, label.repeat(self.num_sample))
    
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0
        bar_grad = 0
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, grad=bar_grad))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            bar_grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True))
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()