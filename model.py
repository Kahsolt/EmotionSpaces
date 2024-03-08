#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import *

from data import HEAD_DIMS

MODELS_TO_WEIGHTS = {
  'resnet50': ResNet50_Weights.IMAGENET1K_V1,
  'resnet101': ResNet101_Weights.IMAGENET1K_V1,
}


class HeadLinear(nn.Module):
  
  def __init__(self, d_in:int, d_out:int):
    super().__init__()

    self.fc = nn.Linear(d_in, d_out)

  def forward(self, x:Tensor) -> Tensor:
    return self.fc(x)
  

class HeadMLP(nn.Module):
  
  def __init__(self, d_in:int, d_out:int, d_hid:int=None):
    super().__init__()

    d_hid = d_hid or ((d_in + d_out) * 2)
    self.mlp = nn.Sequential(
      nn.Linear(d_in, d_hid),
      nn.ReLU(inplace=True),
      nn.Linear(d_hid, d_out),
    )

  def forward(self, x:Tensor) -> Tensor:
    return self.mlp(x)


class MultiTaskResNet(nn.Module):

  def __init__(self, model_type:str='resnet50', d_x:int=32, head_type:str='linear', pretrain:bool=False):
    super().__init__()

    # 预训练的ResNet模型
    model: ResNet = globals()[model_type](weights=MODELS_TO_WEIGHTS[model_type] if pretrain else None)
    # features部分保持不变
    self.fvecs = nn.Sequential(
      model.conv1,
      model.bn1,
      model.relu,
      model.maxpool,
      model.layer1,
      model.layer2,
      model.layer3,
      model.layer4,   # [B, C=2048, H//32, W//32]
      model.avgpool,  # [B, C=2048, 1, 1]
      nn.Flatten(1),  # [B, D=2048]
    )
    del model
    # 将fc部分改造为降维MLP
    self.proj = nn.Sequential(
      nn.Linear(2048, 256), # [B, D=256]
      nn.SiLU(),
      nn.Linear(256, d_x),  # [B, D=32]
    )
    # 各下游任务使用不同的线性投影/逆投影
    head_cls = HeadLinear if head_type == 'linear' else HeadMLP
    self.heads = nn.ModuleDict({
      name: head_cls(d_x, d_out) for name, d_out in HEAD_DIMS.items()
    })
    self.invheads = nn.ModuleDict({
      name: head_cls(HEAD_DIMS[name], d_x) for name in self.heads.keys()
    })

  def forward(self, x:Tensor, head:str) -> Tensor:
    fmap = self.fvecs(x)
    fv = self.proj(fmap)
    o = self.heads[head](fv)
    ifv = self.invheads[head](o)
    return o, fv, ifv

  def infer(self, x:Tensor, head:str) -> Tensor:
    fmap = self.fvecs(x)
    fv = self.proj(fmap)
    o = self.heads[head](fv)
    return o


if __name__ == '__main__':
  model = MultiTaskResNet('resnet50')
  X = torch.zeros([4, 3, 224, 224])
  for head in HEAD_DIMS.keys():
    o, fv, ifv = model(X, head)
    print(f'{head}:', *[tuple(x.shape) for x in [o, fv, ifv]])
