#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam
from torchvision.models.resnet import resnet50, ResNet50_Weights
from lightning import LightningModule, Trainer, seed_everything

torch.set_float32_matmul_precision('medium')

from data import *
from utils import *

BATCH_SIZE = 32
EPOCH = 100
BASE_LR = [2e-6, 4e-5, 1e-4, 1e-4]
LOSS_L = 1
LOSS_W = 10
MODEL_HEADS = {
  'Polar':  2,
  'VA':     2,
  'Ekman':  6,
  'EkmanN': 7,
  'Mikels': 8,
}


class MultiTaskResNet(nn.Module):

  def __init__(self, d_x:int=32, load_base:bool=False):
    super().__init__()

    # 交换空间的向量深度
    print('d_x:', d_x)
    # 预训练的ResNet模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if load_base else None)
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
      nn.LazyLinear(256),   # [B, D=256]
      nn.SiLU(),
      nn.Linear(256, d_x),  # [B, D=32]
    )
    # 各下游任务使用不同的线性投影/逆投影
    self.heads = nn.ModuleDict({
      name: nn.Linear(d_x, d_out) for name, d_out in MODEL_HEADS.items()
    })
    self.invheads = nn.ModuleDict({
      name: nn.LazyLinear(d_x) for name in self.heads.keys()
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


class LitModel(LightningModule):

  def __init__(self, model:MultiTaskResNet):
    super().__init__()

    self.model = model
    self.head = None
    self.is_ldl = None
    self.freeze_modules = []
    self.lr_list = [1e-6, 2e-4, 1e-4, 1e-4]

  def set_mode(self, head:str, is_ldl:bool=False, lr_list:List[float]=None, freeze_modules:List[str]=None):
    self.head = head
    self.is_ldl = is_ldl
    if lr_list:
      if isinstance(lr_list, float): lr_list = [lr_list] * len(self.lr_list)
      assert len(lr_list) == len(self.lr_list)
      self.lr_list = lr_list
    if freeze_modules:
      self.freeze_modules = freeze_modules

  def configure_optimizers(self) -> Optimizer:
    param_groups = [
      {'params': self.model.fvecs   .parameters(), 'lr': self.lr_list[0]} if 'fvecs' not in self.freeze_modules else None,
      {'params': self.model.proj    .parameters(), 'lr': self.lr_list[1]} if 'proj'  not in self.freeze_modules else None,
      {'params': self.model.heads   .parameters(), 'lr': self.lr_list[2]},
      {'params': self.model.invheads.parameters(), 'lr': self.lr_list[3]},
    ]
    return Adam([it for it in param_groups if it])

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr-{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def get_losses(self, batch:Tuple[Tensor], batch_idx:int) -> Tuple[Tensor, Dict[str, float]]:
    x, y = batch
    out, fvec, invfvec = self.model(x, self.head)
    if self.head in ['VA']:
      loss_task = F.mse_loss(out, y)
    elif self.head in ['Polar', 'Ekman', 'EkmanN', 'Mikels']:
      loss_clf = F.cross_entropy(out, y)
      loss_ldl = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean') if self.is_ldl else 0.0
      loss_task = loss_clf + loss_ldl * LOSS_L
    loss_recon = F.mse_loss(invfvec, fvec)
    loss = loss_task + loss_recon * LOSS_W
    if batch_idx % 10 == 0:
      with torch.no_grad():
        log_dict = {
          'l_task': loss_task.item(),
          'l_rec': loss_recon.item(),
          'l_total': loss.item(),
        }
    else: log_dict = None
    return loss, log_dict

  def log_losses(self, log_dict:Dict[str, float], prefix:str='log'):
    self.log_dict({f'{prefix}/{k}': v for k, v in log_dict.items()})

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    loss, log_dict = self.get_losses(batch, batch_idx)
    if log_dict: self.log_losses(log_dict, 'train')
    return loss

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    loss, log_dict = self.get_losses(batch, batch_idx)
    if log_dict: self.log_losses(log_dict, 'valid')
    return loss


def train(model:MultiTaskResNet, dataset_cls:Callable[[Any], Dataset], **kwargs):
  head: str = kwargs.get('head')
  is_ldl: bool = kwargs.get('is_ldl')
  lr_list: List[float] = kwargs.get('lr_list', None)
  freeze_modules: List[str] = kwargs.get('freeze_modules', None)
  batch_size: int = kwargs.get('batch_size', 32)
  epochs: int = kwargs.get('epochs', 10)
  
  seed_everything(42)
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
  }
  trainloader = DataLoader(dataset_cls('train'), batch_size, shuffle=True, **dataloader_kwargs)
  validloader = DataLoader(dataset_cls('valid'), batch_size=1, shuffle=False, **dataloader_kwargs)
  lit = LitModel(model)
  lit.set_mode(head, is_ldl, lr_list, freeze_modules)
  trainer = Trainer(
    max_epochs=epochs,
    precision='16-mixed',
    benchmark=True,
    enable_checkpointing=True,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  DATASET_CONFIGS = [
    # head, is_ldl, dataset_cls
    ('Polar',  False, TweeterI),      # TwitterI
    ('VA',     False, Emotion6VA),    # Emotion6, OASIS, GAPED
    ('Ekman',  True,  Emotion6Dim6),  # Emotion6
    ('Ekman',  False, FER2013),       # FER-2013
    ('EkmanN', True,  Emotion6Dim7),  # Emotion6
    ('EkmanN', False, FER2013),       # FER-2013
    ('Mikels', True,  Abstract),      # Abstract
    ('Mikels', False, FI),            # ArtPhoto, FI, EmoSet-118K
  ]

  model = MultiTaskResNet(load_base=True)
  for head, is_ldl, dataset_cls in DATASET_CONFIGS:
    train(model, dataset_cls, head, is_ldl)
