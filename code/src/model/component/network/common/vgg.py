from torch import nn
import torch
from torchvision import models


class VGG19PyTorch(nn.Module):
    # Note: use this model instead of the model from deep video exemplar-based colorization
    def __init__(self):
        super().__init__()
        pretrained_vgg19_model = models.vgg19(weights=models.VGG19_Weights)
        blocks = []
        blocks.append(pretrained_vgg19_model.features[:4].eval())
        blocks.append(pretrained_vgg19_model.features[4:9].eval())
        blocks.append(pretrained_vgg19_model.features[9:18].eval())
        blocks.append(pretrained_vgg19_model.features[18:27].eval())
        blocks.append(pretrained_vgg19_model.features[27:36].eval())

        self.layer_names = ["r1_2", "r2_2", "r3_4", "r4_4", "r5_4"]
        self.blocks = nn.ModuleList(blocks)
        for n, p in self.blocks.named_parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        # x must be unprocessed RGB tensor with range [0, 1]
        out = {}
        x = (x - self.mean) / self.std
        for i, block in enumerate(self.blocks):
            x = block(x)
            out[self.layer_names[i]] = x
        return out


class VGG19PerceptualNetwork(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
