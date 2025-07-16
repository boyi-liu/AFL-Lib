import torch
import torch.nn as nn
import torchvision.models as models


class Multimodal_Resnet18(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        self.fusion_mode = args.fusion
        self.num_classes = num_classes
        self.feature_dim = 512

        if 'cremad' in args.dataset:
            self.encoders = nn.ModuleList([ImgEncoderResnet18(),
                                           AudioEncoderResnet18(), ])

        self.modality_num = len(self.encoders)

        if self.fusion_mode == 'concat':
            self.head = ConcatHead(num_classes, self.modality_num, feature_dim=self.feature_dim)
        elif self.fusion_mode == 'latesum':
            self.head = LateSumHead(num_classes, self.modality_num, feature_dim=self.feature_dim)

    def forward(self, x_list: list[torch.Tensor]):
        if len(x_list) != self.modality_num:
            raise ValueError(
                    f"Multimodal inputs 'x' list length ({len(x_list)}) must match modality number ({self.modality_num}).")
        if all(x is None for x in x_list):
            raise ValueError("Multimodal inputs 'x' can not be all of 'None'. At least one modality must be provided.")

        features = [encoder(x) for x, encoder in zip(x_list, self.encoders)]
        x = self.head(features)
        return features, x


# ==========================
# ====== base modules ======
# ==========================

class ImgEncoderResnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None, num_classes=10)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        if x is None: return None
        x = [self.resnet(_x) for _x in torch.chunk(x, 3, dim=2)]
        x = torch.stack(x).mean(dim=0, keepdim=False)

        return x


class AudioEncoderResnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None, num_classes=10)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):  # Add channel dimension
        if x is None: return None
        x = self.resnet(x)
        return x


class Head(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(Head, self).__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    # def forward(self, x):
    #     return self.classifier(x)


class LateSumHead(Head):
    def __init__(self, num_classes, modality_num, feature_dim):
        super().__init__(num_classes, feature_dim)
        self.classifier = nn.ModuleList([
            nn.Linear(feature_dim, num_classes),
            nn.Linear(feature_dim, num_classes),
        ])
        self.modality_num = modality_num

    def forward(self, x_list: list[torch.Tensor]):
        outputs = []
        for x, liner in zip(x_list, self.classifier):
            if x is not None:
                outputs.append(liner(x))
        x = torch.sum(torch.stack(outputs), dim=0)
        return x


class ConcatHead(Head):
    def __init__(self, num_classes, modality_num, feature_dim=512):
        super().__init__(num_classes, feature_dim)
        self.classifier = nn.Linear(feature_dim * modality_num, num_classes)

    def forward(self, x_list: list[torch.Tensor]):
        for x in x_list:
            if x is not None:
                ref_tensor = x
                break
        for i in range(len(x_list)):
            if x_list[i] is None:
                x_list[i] = torch.zeros_like(ref_tensor)
        x = torch.cat(x_list, dim=1)
        x = self.classifier(x)
        return x


def multimodalresnet18_cremad(args, **kwargs):
    return Multimodal_Resnet18(args, **kwargs)
