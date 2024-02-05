# Model components

from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.models as tvm

BACKBONE_DIR = "../pretrained_weights"
torch.hub.set_dir(BACKBONE_DIR)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_backbone(
    base_architecture: str = "r2plus1d_18", pretrained: bool = False
):
    if base_architecture == "r2plus1d_18":
        backbone_weights = "KINETICS400_V1" if pretrained else None
        backbone = tvm.video.r2plus1d_18(weights=backbone_weights)
        output_dim = backbone.head.in_features
        # remove the fc layer only, preserving 512-dim latent space
        backbone.fc = Identity()
        output_dim = 

    elif base_architecture == "swin3d_t":
        backbone_weights = "KINETICS400_V1" if pretrained else None
        backbone = tvm.video.swin3d_t(weights=backbone_weights)
        output_dim = backbone.head.in_features
        assert output_dim == 768
        backbone.head = Identity()
    else:
        raise NotImplementedError()

    return backbone, output_dim
    
if __name__ == __main__:
    backbone = tvm.video.r2plus1d_18(weights=None)
    print(backbone)
