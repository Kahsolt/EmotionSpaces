#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam
from torch.optim.lr_scheduler import CyclicLR
from torchvision.models.resnet import resnet50, ResNet50_Weights
from lightning import LightningModule, Trainer, seed_everything

torch.set_float32_matmul_precision('medium')

from data import *
from utils import *

BATCH_SIZE = 32
EPOCH = 100
BASE_LR = [2e-6, 4e-5, 1e-4, 1e-4]
MAX_LR = [2e-5, 4e-4, 1e-3, 1e-3]
MOMENTUM = 0.9
LOSS_W = 10


class MultiTaskResNet(nn.Module):

  def __init__(self, load_base:bool=False):
    super().__init__()

    # 交换空间的向量深度
    dim = 32
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
      model.layer4,     # [B, C=2048, H//32, W//32]
      model.avgpool,    # [B, C=2048, 1, 1]
      nn.Flatten(start_dim=1),
    )
    del model
    # 将fc部分改造为降维MLP
    self.proj = nn.Sequential(
      nn.LazyLinear(256),
      nn.SiLU(),
      nn.Linear(256, dim),    # [B, D=32]
    )
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

  def set_mode(self, head:str, is_ldl:bool=False):
    self.head = head
    self.is_ldl = is_ldl

  def configure_optimizers(self) -> Optimizer:
    param_groups = [
      {'params': self.model.fvecs   .parameters(), 'lr': BASE_LR[0]},              
      {'params': self.model.proj    .parameters(), 'lr': BASE_LR[1]},              
      {'params': self.model.heads   .parameters(), 'lr': BASE_LR[2]},
      {'params': self.model.invheads.parameters(), 'lr': BASE_LR[3]},
    ]
    #optim = Adam(param_groups)
    optim = SGD(param_groups, momentum=MOMENTUM)
    sched = CyclicLR(optim, base_lr=BASE_LR, max_lr=MAX_LR, step_size_up=1000, step_size_down=2000, mode='triangular2')
    return {
      'optimizer': optim,
      'lr_scheduler': sched,
    }

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr-{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, y = batch
    out, fvec, invfvec = self.model(x, self.head)
    if self.head in ['VA']:
      loss_task = F.mse_loss(out, y)
    elif self.head in ['Polar', 'Ekman', 'EkmanN', 'Mikels']:
      loss_task = F.cross_entropy(out, y)
      if self.is_ldl:
        loss_task += F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean')
    loss_recon = F.mse_loss(invfvec, fvec)
    loss = loss_task + loss_recon * LOSS_W
    if batch_idx % 10 == 0:
      with torch.no_grad():
        self.log_dict({
          'train/l_task': loss_task.item(),
          'train/l_rec': loss_recon.item(),
          'train/l_total': loss.item(),
        })
    return loss

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, y = batch
    out, fvec, invfvec = self.model(x, self.head)
    if self.head in ['VA']:
      loss_task = F.mse_loss(out, y)
    elif self.head in ['Polar', 'Ekman', 'EkmanN', 'Mikels']:
      loss_task = F.cross_entropy(out, y)
      if self.is_ldl:
        loss_task += F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean')
    loss_recon = F.mse_loss(invfvec, fvec)
    loss = loss_task + loss_recon * LOSS_W
    if batch_idx % 10 == 0:
      with torch.no_grad():
        self.log_dict({
          'valid/l_task': loss_task.item(),
          'valid/l_rec': loss_recon.item(),
          'valid/l_total': loss.item(),
        })


def train():
  seed_everything(42)

  sel = 0
  if sel == 0:
    head = 'VA'
    is_ldl = False
    trainset = Emotion6VA('train')
    validset = Emotion6VA('test')
  elif sel == 1:
    head = 'Ekman'
    is_ldl = True
    trainset = Emotion6Dim6('train')
    validset = Emotion6Dim6('test')

  kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
  }
  trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
  validloader = DataLoader(validset, batch_size=1, shuffle=False, **kwargs)
  model = MultiTaskResNet(load_base=True)
  lit = LitModel(model)
  lit.set_mode(head, is_ldl)
  trainer = Trainer(
    max_epochs=100,
    precision='16-mixed',
    benchmark=True,
  )
  trainer.fit(lit, trainloader, validloader)


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
