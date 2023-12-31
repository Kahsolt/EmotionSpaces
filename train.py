#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam
from torchvision.models.resnet import resnet50, ResNet50_Weights
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything

torch.set_float32_matmul_precision('medium')

from data import *
from utils import *


class MultiTaskResNet(nn.Module):

  def __init__(self, load_base:bool=False):
    super().__init__()

    # 交换空间的向量深度
    dim = 32
    # 预训练的ResNet模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if load_base else None)
    self.fvecs = nn.Sequential(
      # features部分保持不变
      model.conv1,
      model.bn1,
      model.relu,
      model.maxpool,
      model.layer1,
      model.layer2,
      model.layer3,
      model.layer4,     # [B, C=2048, H//32, W//32]
      model.avgpool,    # [B, C=2048, 1, 1]
      nn.Flatten(start_dim=1),
      # 将fc部分改造为降维MLP
      nn.LazyLinear(256),
      nn.SiLU(),
      nn.Linear(256, dim),
    )
    del model
    # 各下游任务使用不同的线性投影/逆投影
    self.heads = nn.ModuleDict({
      'Polar':  nn.Linear(dim, 2),
      'VA':     nn.Linear(dim, 2),
      'Ekman':  nn.Linear(dim, 6),
      'EkmanN': nn.Linear(dim, 7),
      'Mikels': nn.Linear(dim, 8),
    })
    self.invheads = nn.ModuleDict({
      name: nn.LazyLinear(dim) for name in self.heads.keys()
    })

  @property
  def head_names(self) -> List[str]:
    return [name for name, _ in self.heads.named_modules()]

  def forward(self, x:Tensor, head:str) -> Tensor:
    fvec = self.fvecs(x)
    out = self.heads[head](fvec)
    invfvec = self.invheads[head](out)
    return out, fvec, invfvec

  def infer(self, x:Tensor, head:str) -> Tensor:
    fvec = self.fvecs(x)
    out = self.heads[head](fvec)
    return out


class LitModel(LightningModule):

  def __init__(self, model:MultiTaskResNet):
    super().__init__()

    self.model = model
    self.head = 'VA'
    self.data = LightningDataModule.from_datasets(
      train_dataset=Emotion6VA('train'),
      val_dataset=Emotion6VA('test'),
      batch_size=32,
    )

  def configure_optimizers(self) -> Optimizer:
    param_groups = [
      {'params': self.model.fvecs.parameters(), 'lr': 1e-5},              
      {'params': self.model.heads.parameters(), 'lr': 1e-3},
    ]
    return Adam(param_groups, lr=1e-3)

  def training_step(self, batch:Tuple[Tensor]) -> Tensor:
    x, y = batch
    breakpoint()
    out, fvec, invfvec = self.model(x, y, self.head)
    # TODO: what loss to use
    loss_task = F.mse_loss(out, y)
    loss_task = F.binary_cross_entropy(out, y)
    loss_task = F.binary_cross_entropy_with_logits(out, y)
    loss_task = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean')
    loss_recon = F.mse_loss(invfvec, fvec)
    loss = loss_task + loss_recon * 0.1
    return loss


def train():
  seed_everything(42)

  model = MultiTaskResNet(load_base=True)
  lit = LitModel(model)
  trainer = Trainer(
    max_epochs=100,
    precision='16-mixed',
    benchmark=True,
  )
  trainer.fit(lit, datamodule=lit.data)


@torch.inference_mode()
def infer():
  fp = './lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt'
  model = MultiTaskResNet()
  lit = LitModel.load_from_checkpoint(fp, model=model)
  model = lit.model.eval()
  x = torch.rand([1, 3, 224, 224])
  pred = model(x, 'VA')
  print(pred)


if __name__ == '__main__':
  train()
