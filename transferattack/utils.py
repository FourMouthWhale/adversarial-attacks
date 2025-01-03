import torch
import torch.utils
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import os

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = ['resnet18', 'resnet101', 'resnet50_32x4d']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

def load_pretrained_model(cnn_model=[], vit_model=[]):
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
    for model_name in vit_model:
        yield model_name, timm.create_model(model_name, pretrained=True)

    
def warp_model(model):
    if hasattr(model, 'default_cfg'):
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError
        

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'label.csv'))

        if eval:
            self.data_dir = output_dir
            print("=> Eval mode: evaluating on {}".format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print("=> Train mode: training on {}".format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())
    
    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        
        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename 

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], 
                                                 dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted', targeted=True, eval=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break