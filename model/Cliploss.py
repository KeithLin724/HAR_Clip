import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale, targets):
        """
        image_features: [B, D]
        text_features: [C, D]
        logit_scale: scalar
        targets: [B] long tensor
        """
        logits_per_image = logit_scale * image_features @ text_features.T  # [B, C]
        return F.cross_entropy(logits_per_image, targets)