import torch

from ..utils import *
from ..attack import Attack
import math

class AIFGSM(Attack):
    def __init__(self, model_name, attack='AI-FGSM', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None, beta_1=0.99, beta_2=0.99, lam=1.3, mu_1=1.5, mu_2=1.9, **kwargs):
        super().__init__(model_name, attack, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lam = lam
        self.mu_1 = mu_1
        self.mu_2 = mu_2

    def get_alpha(self, T, t_):
        res = 0
        for t in range(T):
            res += (1-self.beta_1**(t+1))/math.sqrt(1-self.beta_2**(t+1))
        return self.epsilon / res * (1-self.beta_1**(t_+1))/math.sqrt(1-self.beta_2**(t_+1))

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.tanh(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta
    
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0
        v = 0
        for _ in range(self.epoch):
            logits = self.get_logits(data + delta)
            loss = self.get_loss(logits, label)
            self.model.zero_grad()
            grad = self.get_grad(loss, delta)

            momentum = momentum + self.mu_1 * grad
            v = v + self.mu_2 * grad * grad
            alpha = self.get_alpha(self.epoch, _)
            delta = self.update_delta(delta, data, self.lam * momentum / (torch.sqrt(v) + 1e-20), alpha)

        return delta.detach()