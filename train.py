#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/16

import sys
from argparse import ArgumentParser

from torch.optim import Optimizer, SGD, Adam
from torchvision.models.resnet import *
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy

from data import *
from model import *
from utils import *

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


class LitModel(LightningModule):

  def __init__(self, model:MultiTaskResNet):
    super().__init__()

    self.model = model
    # ↓↓ training specified ↓↓
    self.args = None
    self.head = None
    self.is_ldl = None
    self.lr_list: List[float] = [1e-6, 1e-4, 2e-4]
    self.acc_train = None
    self.acc_valid = None
    self.mse_train = None
    self.mse_valid = None

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
    if self.is_clf:
      self.acc_train = MulticlassAccuracy(n_class)
      self.acc_valid = MulticlassAccuracy(n_class)
    else:
      self.mse_train = MeanSquaredError()
      self.mse_valid = MeanSquaredError()

  def configure_optimizers(self) -> Optimizer:
    param_groups = [
      {'params': self.model.fvecs   .parameters(), 'lr': self.lr_list[0]},
      {'params': self.model.proj    .parameters(), 'lr': self.lr_list[1]},
      {'params': self.model.heads   .parameters(), 'lr': self.lr_list[2]},
      {'params': self.model.invheads.parameters(), 'lr': self.lr_list[2]},
    ]
    return Adam([it for it in param_groups if it['lr'] > 0], weight_decay=1e-5)

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    #if batch_idx % 10 == 0:
    #  self.log_dict({f'lr/group-{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def forward_step(self, batch:Tuple[Tensor], prefix:str) -> Tensor:
    x, y = batch
    out, fvec, invfvec = self.model(x, self.head)

    y_lbl = torch.argmax(y, dim=-1) if self.is_ldl else y
    is_train = prefix == 'train'
    if self.is_clf:
      loss_clf = F.cross_entropy(out, y_lbl)
      loss_ldl = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean') if self.is_ldl else 0.0
      loss_task = loss_clf + loss_ldl * self.args.loss_w_ldl
      metric = getattr(self, f'acc_{prefix}')
      metric(out, y_lbl)
      self.log(f'{prefix}/acc', metric, on_step=is_train, on_epoch=True)
    else:
      loss_task = F.mse_loss(out, y)
      metric = getattr(self, f'mse_{prefix}')
      metric(out, y_lbl)
      self.log(f'{prefix}/mse', metric, on_step=is_train, on_epoch=True)

    loss_recon = F.mse_loss(invfvec, fvec)
    loss: Tensor = loss_task + loss_recon * self.args.loss_w_recon

    with torch.no_grad():
      log_dict = {
        'l_clf': locals().get('loss_clf').item() if locals().get('loss_clf') else None,
        'l_ldl': locals().get('loss_ldl').item() if locals().get('loss_ldl') else None,
        'l_task': loss_task.item(),
        'l_rec': loss_recon.item(),
        'l_total': loss.item(),
      }
      log_dict = {k: v for k, v in log_dict.items() if v is not None}
      self.log_dict({f'{prefix}/{k}': v for k, v in log_dict.items()})
    return loss

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    return self.forward_step(batch, 'train')

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    return self.forward_step(batch, 'valid')


def train(args):
  seed_everything(args.seed)
  print('>> cmd:', ' '.join(sys.argv))
  print('>> args:', vars(args))

  ''' Data '''
  assert args.dataset, 'must provide --dataset'
  dataset_cls: Callable[[Any], Dataset] = globals()[args.dataset]
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainloader = DataLoader(dataset_cls('train'), args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(dataset_cls('valid'), batch_size=1,    shuffle=False, drop_last=False, **dataloader_kwargs)
  n_class = trainloader.dataset.n_class

  ''' Model & Optim '''
  model = MultiTaskResNet(args.model, args.d_x, args.head, pretrain=args.load is None)
  lit = LitModel(model)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args, n_class)

  ''' Train '''
  metric_name = 'valid/acc' if lit.is_clf else 'valid/mse'
  metric_mode = 'max' if lit.is_clf else 'min'
  checkpoint_callback = ModelCheckpoint(monitor=metric_name, mode=metric_mode)
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    callbacks=[checkpoint_callback],
    limit_val_batches=1000,
  )
  trainer.fit(lit, trainloader, validloader)


def get_parser():
  parser = ArgumentParser()
  parser.add_argument('-L', '--load',  type=Path, help='load pretrained weights')
  parser.add_argument('-M', '--model', default='resnet50', choices=['resnet50', 'resnet101'])
  parser.add_argument('-H', '--head', default='linear', choices=['linear', 'mlp'])
  parser.add_argument('-X', '--d_x', default=32, type=int, help='Xspace dim')
  parser.add_argument('-B', '--batch_size', type=int, default=32)
  parser.add_argument('-E', '--epochs',     type=int, default=10)
  parser.add_argument('-lr', '--lr_list', nargs='+', type=eval, default=2e-4, help='lr list for each part: fvecs/proj/heads/invheads')
  parser.add_argument('--loss_w_ldl',   type=float, default=1,  help='loss weight for ldl (kl_div loss)')
  parser.add_argument('--loss_w_recon', type=float, default=10, help='loss weight for x-space reconstruction')
  parser.add_argument('--seed', type=int, default=114514)
  return parser


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('-D', '--dataset', default='Emotion6Dim7', choices=DATASETS)
  args = parser.parse_args()

  train(args)
