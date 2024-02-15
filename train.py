#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

from __future__ import annotations

import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam
from torchvision.models.resnet import resnet50, ResNet50_Weights
from lightning import LightningModule, Trainer, seed_everything
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy

torch.set_float32_matmul_precision('medium')

from data import *
from utils import *

HEAD_DIMS = {
  # 'head_type': dim
  'Mikels': 8,
  'EkmanN': 7,
  'Ekman':  6,
  'VA':     2,
  'Polar':  2,
}
HEAD_DATASET_CONFIGS = {
  # 'head_type': {'dataset_cls': is_ldl}
  'Mikels': {
    'EmoSet': False,
    'FI': False,
    'ArtPhoto': False,
    'Abstract': True,
  },
  'EkmanN': {
    'Emotion6Dim7': True,
    'FER2013': False,
  },
  'Ekman': {
    'Emotion6Dim6': True,
    'FER2013': False,
  },
  'VA': {
    'Emotion6VA': False,
    'OASIS': False,
    'GAPED': False,
  },
  'Polar': {
    'TweeterI': True,
  }
}
DATASET_TO_HEAD_TYPE = {ds: head for head, ds_cfgs in HEAD_DATASET_CONFIGS.items() for ds in ds_cfgs}
DATASETS = list(DATASET_TO_HEAD_TYPE.keys())


class MultiTaskResNet(nn.Module):

  def __init__(self, d_x:int=32, pretrain:bool=False):
    super().__init__()

    # 交换空间的向量深度
    print('d_x:', d_x)
    # 预训练的ResNet模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrain else None)
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
      name: nn.Linear(d_x, d_out) for name, d_out in HEAD_DIMS.items()
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
    # ↓↓ training specified ↓↓
    self.args = None
    self.head = None
    self.is_ldl = None
    self.freeze_modules: List[str] = []
    self.lr_list: List[float] = [1e-6, 1e-4, 2e-4, 2e-4]
    self.acc = None
    self.mse = None

  def setup_train_args(self, args, n_class:int):
    self.args = args
    self.head = DATASET_TO_HEAD_TYPE[args.dataset]
    self.is_clf = self.head not in ['VA']
    self.is_ldl = HEAD_DATASET_CONFIGS[self.head][args.dataset]
    if args.lr_list:
      lr_list: List[float] = args.lr_list
      if isinstance(lr_list, float): lr_list = [lr_list] * len(self.lr_list)
      if len(lr_list) == 1: lr_list *= len(self.lr_list)
      assert len(lr_list) == len(self.lr_list)
      self.lr_list = lr_list
    if args.freeze_modules:
      self.freeze_modules = args.freeze_modules
    self.acc = MulticlassAccuracy(n_class)
    self.mse = MeanSquaredError()

  def configure_optimizers(self) -> Optimizer:
    param_groups = [
      {'params': self.model.fvecs   .parameters(), 'lr': self.lr_list[0]} if 'fvecs' not in self.freeze_modules else None,
      {'params': self.model.proj    .parameters(), 'lr': self.lr_list[1]} if 'proj'  not in self.freeze_modules else None,
      {'params': self.model.heads   .parameters(), 'lr': self.lr_list[2]},
      {'params': self.model.invheads.parameters(), 'lr': self.lr_list[3]},
    ]
    return Adam([it for it in param_groups if it], weight_decay=1e-5)

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr-{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def get_losses(self, batch:Tuple[Tensor], batch_idx:int) -> Tuple[Tensor, Dict[str, float]]:
    x, y = batch
    out, fvec, invfvec = self.model(x, self.head)
    if self.is_clf:
      loss_clf = F.cross_entropy(out, y)
      loss_ldl = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean') if self.is_ldl else 0.0
      loss_task = loss_clf + loss_ldl * self.args.loss_w_ldl
    else:
      loss_task = F.mse_loss(out, y)
    loss_recon = F.mse_loss(invfvec, fvec)
    loss: Tensor = loss_task + loss_recon * self.args.loss_w_recon

    if batch_idx % 10 == 0:
      with torch.no_grad():
        _ = self.acc(out, y) if self.is_clf else self.mse(out, y)
        log_dict = {
          'l_clf': locals().get('loss_clf').item() if locals().get('loss_clf') else None,
          'l_ldl': locals().get('loss_ldl').item() if locals().get('loss_ldl') else None,
          'l_task': loss_task.item(),
          'l_rec': loss_recon.item(),
          'l_total': loss.item(),
        }
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
    else: log_dict = None
    return loss, log_dict

  def on_train_epoch_end(self) -> None:
    if self.is_clf:
      self.log('train/acc/epoch', self.acc, on_step=False, on_epoch=True)
    else:
      self.log('train/mse/epoch', self.mse, on_step=False, on_epoch=True)

  def on_validation_epoch_end(self) -> None:
    if self.is_clf:
      self.log('valid/acc/epoch', self.acc, on_step=False, on_epoch=True)
    else:
      self.log('valid/mse/epoch', self.mse, on_step=False, on_epoch=True)

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


def train(args):
  seed_everything(args.seed)
  print('>> cmd:', ' '.join(sys.argv))
  print('>> args:', vars(args))

  ''' Data '''
  dataset_cls: Callable[[Any], Dataset] = globals()[args.dataset]
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainloader = DataLoader(dataset_cls('train'), args.batch_size, shuffle=True,  **dataloader_kwargs)
  validloader = DataLoader(dataset_cls('valid'), batch_size=1,    shuffle=False, **dataloader_kwargs)
  n_class = trainloader.dataset.n_class

  ''' Model & Optim '''
  model = MultiTaskResNet(pretrain=True)
  lit = LitModel(model)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args, n_class)

  ''' Train '''
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    enable_checkpointing=True,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--load',   type=Path, help='load pretrained weights')
  parser.add_argument('-D', '--dataset', required=True, choices=DATASETS)
  parser.add_argument('-B', '--batch_size', type=int, default=32)
  parser.add_argument('-E', '--epochs',     type=int, default=10)
  parser.add_argument('-lr', '--lr_list', nargs='+', type=eval, default=2e-4, help='lr list for each part: fvecs/proj/heads/invheads')
  parser.add_argument('-fm', '--freeze_modules', nargs='+', help='freeze weights of backbone parts: fvecs/proj')
  parser.add_argument('--loss_w_ldl',   type=float, default=1,  help='loss weight for ldl (kl_div loss)')
  parser.add_argument('--loss_w_recon', type=float, default=10, help='loss weight for x-space reconstruction')
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
