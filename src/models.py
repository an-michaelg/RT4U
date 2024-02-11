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
        output_dim = backbone.fc.in_features
        assert output_dim == 512
        # remove the fc layer only, preserving 512-dim latent space
        backbone.fc = Identity()

    elif base_architecture == "swin3d_t":
        backbone_weights = "KINETICS400_V1" if pretrained else None
        backbone = tvm.video.swin3d_t(weights=backbone_weights)
        output_dim = backbone.head.in_features
        assert output_dim == 768
        backbone.head = Identity()
        
    elif base_architecture == "resnet_18":
        backbone_weights = "IMAGENET1K_V1" if pretrained else None
        backbone = tvm.resnet18(weights=backbone_weights)
        output_dim = backbone.fc.in_features
        assert output_dim == 512
        backbone.fc = Identity()
        
    else:
        raise NotImplementedError()

    print(f"Model backbone initialized {base_architecture}")
    return backbone, output_dim
    
if __name__ == "__main__":
    backbone, output_dim = get_backbone("resnet_18")
    print(backbone)
