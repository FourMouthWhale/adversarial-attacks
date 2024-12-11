import torch
from ..utils import *
from ..attack import Attack
import torch.nn.functional as F

class PIFGSM(Attack):
    def __init__(self, model_name, attack='PI-FGSM', epsilon=16.0/255, alpha=1.6/255, epoch=10, decay=0, 
                 kern_size=3, gamma=16.0, beta=10.0, targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None):
        super().__init__(model_name, attack, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.kern_size = kern_size
        self.gamma = gamma / 255.0
        self.beta = beta
        self.model = self.load_model(model_name)
        self.device = next(self.model.parameters()).device if device is None else device

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).to(self.device)
        return stack_kern, kern_size // 2
    
    def project_noise(self, x, stack_kern, padding_size):
        x = F.conv2d(x, stack_kern, padding = (padding_size, padding_size), groups=3)
        return x
    
    def update_delta(self, delta, data, grad, alpha, projection, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign() + projection, -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha + projection).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta
    
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        delta.requires_grad = True

        stack_kern, padding_size = self.project_kern(self.kern_size)

        momentum, amplification = 0.0, 0.0

        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data + delta))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            amplification += self.beta * self.alpha * momentum.sign()
            cut_noise = torch.clamp(abs(amplification) - self.epsilon, 0, 10000.0) * torch.sign(amplification)
            projection = self.gamma * torch.sign(self.project_noise(cut_noise, stack_kern, padding_size))
            amplification += projection
            delta = self.update_delta(delta, data, momentum, self.beta * self.alpha, projection)
        
        return delta.detach()