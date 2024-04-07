#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import *
from torchvision.models.vision_transformer import *
from torchvision.models.mobilenet import *
from torchvision.models.squeezenet import *

from data import HEAD_DIMS, is_clf

MODELS_TO_WEIGHTS = {
  'resnet18': ResNet18_Weights.IMAGENET1K_V1,
  'resnet34': ResNet34_Weights.IMAGENET1K_V1,
  'resnet50': ResNet50_Weights.IMAGENET1K_V1,
  'resnet101': ResNet101_Weights.IMAGENET1K_V1,
  'vit_b_16': ViT_B_16_Weights.IMAGENET1K_V1,
  'vit_b_32': ViT_B_32_Weights.IMAGENET1K_V1,
  'mobilenet_v2': MobileNet_V2_Weights.IMAGENET1K_V1,
}


class PretrainedBackbone(nn.Module):

  def __init__(self, name:str, pretrain:bool=True):
    super().__init__()

    self.model: Backbone = globals()[name](weights=MODELS_TO_WEIGHTS[name] if pretrain else None)
    self.dim_out = None

  @property
  def d_out(self) -> int:
    return self.dim_out

  def forward(self, x:Tensor) -> Tensor:
    raise NotImplementedError


class ResnetBackbone(PretrainedBackbone):

  def __init__(self, name:str='resnet50', pretrain:bool=True):
    super().__init__(name, pretrain)

    self.model: ResNet
    self.dim_out = self.model.fc.in_features
    del self.model.fc

  def forward(self, x:Tensor) -> Tensor:
    assert ResNet.forward
    self = self.model

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class ViTBackbone(PretrainedBackbone):

  def __init__(self, name:str='vit_b_16', pretrain:bool=True):
    super().__init__(name, pretrain)

    self.model: VisionTransformer
    self.dim_out = self.model.heads[0].in_features
    del self.model.heads

  def forward(self, x:Tensor) -> Tensor:
    assert VisionTransformer.forward
    self = self.model

    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = self.encoder(x)
    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    return x


class MobileNetBackbone(PretrainedBackbone):

  def __init__(self, name:str='mobilenet_v2', pretrain:bool=True):
    super().__init__(name, pretrain)

    self.model: MobileNetV2
    self.dim_out = self.model.last_channel
    del self.model.classifier

  def forward(self, x:Tensor) -> Tensor:
    assert MobileNetV2.forward
    self = self.model

    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    return x


class LinearHead(nn.Module):
  
  def __init__(self, d_in:int, d_out:int, bias:bool=True):
    super().__init__()

    self.fc = nn.Linear(d_in, d_out, bias)

  def forward(self, x:Tensor) -> Tensor:
    return self.fc(x)


class MLPHead(nn.Module):
  
  def __init__(self, d_in:int, d_out:int, bias:bool=True, d_hid:int=None):
    super().__init__()

    d_hid = d_hid or round((d_in + d_out) * 1.5)
    self.mlp = nn.Sequential(
      nn.Linear(d_in, d_hid, bias),
      nn.SiLU(inplace=True),
      nn.Linear(d_hid, d_out, bias),
    )

  def forward(self, x:Tensor) -> Tensor:
    return self.mlp(x)


BACKBONE_CLASSES = {
  'resnet18': ResnetBackbone,
  'resnet34': ResnetBackbone,
  'resnet50': ResnetBackbone,
  'resnet101': ResnetBackbone,
  'vit_b_16': ViTBackbone,
  'vit_b_32': ViTBackbone,
  'mobilenet_v2': MobileNetBackbone,
}
HEAD_CLASSES = {
  'linear': LinearHead,
  'mlp': MLPHead,
}

Backbone = Union[ResnetBackbone, ViTBackbone, MobileNetBackbone]
Head = Union[LinearHead, MLPHead]

# https://math.stackexchange.com/questions/2786600/invert-the-softmax-function
inv_softmax = lambda x: torch.log(x + 1e-15) + torch.log(torch.exp(x).sum())


class MultiTaskNet(nn.Module):

  def __init__(self, backbone_type:str='resnet50', head_type:str='linear', d_x:int=32, use_bias:bool=True, pretrain:bool=False):
    super().__init__()

    # 预训练的backbone作特征提取器
    self.backbone: Backbone = BACKBONE_CLASSES[backbone_type](backbone_type, pretrain)
    # 线性投影到交换空间
    self.proj = nn.Linear(self.backbone.d_out, d_x)   # 2048 => 32
    # 各下游任务使用不同的head/invhead
    head_cls = HEAD_CLASSES[head_type]
    self.heads = nn.ModuleDict({
      name: head_cls(d_x, d_out, use_bias) for name, d_out in HEAD_DIMS.items()
    })
    self.invheads = nn.ModuleDict({
      name: head_cls(HEAD_DIMS[name], d_x, use_bias) for name in self.heads.keys()
    })

  @property
  def d_x(self):
    return self.proj.out_features

  def forward(self, x:Tensor, head:str) -> Tensor:
    fv = self.backbone(x)         # feature vector
    xv = self.proj(fv)            # xspace vector
    o = self.heads[head](xv)      # outputs/logits
    ixv = self.invheads[head](o)  # inverted xspace vector
    return o, xv, ixv

  @torch.inference_mode()
  def infer(self, x:Tensor, head:str) -> Tensor:
    ''' predict image: img -> ev '''
    fv = self.backbone(x)
    xv = self.proj(fv)
    o = self.heads[head](xv)
    return F.softmax(o, dim=-1) if is_clf(head) else o

  @torch.inference_mode()
  def ev_to_xv(self, ev:Tensor, head:str) -> Tensor:
    ''' unproject Espace to Xspace: ev -> xv '''
    assert ev.shape[-1] == HEAD_DIMS[head]
    o = inv_softmax(ev) if is_clf(head) else ev
    return self.invheads[head](o)

  @torch.inference_mode()
  def xv_to_ev(self, xv:Tensor, head:str) -> Tensor:
    ''' project Xspace to Espace: xv -> ev '''
    assert xv.shape[-1] == self.d_x
    o = self.heads[head](xv)
    return F.softmax(o, dim=-1) if is_clf(head) else o


if __name__ == '__main__':
  model = MultiTaskNet('resnet50', 'linear', d_x=32)
  print(model)
  X = torch.zeros([4, 3, 224, 224])
  for head in HEAD_DIMS.keys():
    o, xv, ixv = model(X, head)
    print(f'{head}:', *[tuple(x.shape) for x in [o, xv, ixv]])
