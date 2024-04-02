#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/08

# 训练多任务模型

import sys
from argparse import ArgumentParser, Namespace

from torch.optim import Optimizer, Adam
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy

from data import *
from model import *
from utils import *


class MixedDataset(Dataset):

  def __init__(self, dataset_names:List[str], split:str, n_batch:int=200, batch_size:int=32):
    super().__init__()

    self.dataset_names = dataset_names
    self.split = split
    self.batch_size = batch_size
    self.n_batch = n_batch
    self.datasets: List[BaseDataset] = [globals()[name](split) for name in dataset_names]

  @property
  def n_datasets(self):
    return len(self.datasets)

  def __len__(self) -> int:
    return self.n_datasets * self.n_batch * self.batch_size

  def __getitem__(self, idx:int):
    # 连续的bs个样本必须属于同一个子数据集
    # 宏batch编号 (每个子数据集都提供ibatch个batch)
    ibatch = idx // (self.n_datasets * self.batch_size)
    # 宏batch内偏移
    ibatch_offset = idx % (self.n_datasets * self.batch_size)
    # 子数据集编号
    idataset = ibatch_offset // self.batch_size
    dataset = self.datasets[idataset]
    # 样本在子数据集batch中的偏移
    ibatch_dataset_offset = ibatch_offset % self.batch_size
    # 该样本在子数据集全体中的顺序编号
    idx_in_dataset = ibatch_dataset_offset + ibatch * self.batch_size
    # 若原数据集太小，重复滚动使用
    idx_with_cyclic = idx_in_dataset % len(dataset)
    head_id = HEAD_NAMES.index(dataset.head.value)  # str => id
    return *dataset[idx_with_cyclic], head_id, dataset.is_ldl


class LitModel(LightningModule):

  def __init__(self, model:MultiTaskNet, args:Namespace=None):
    super().__init__()

    assert isinstance(model, MultiTaskNet), f'>> model must be MultiTaskNet type, but got: {type(model)}'
    self.model = model

    # ↓↓ training specified ↓↓
    if args: self.save_hyperparameters(args)
    self.args = args
    self.lr_list: List[float] = [1e-6, 2e-4, 2e-4]
    self.metrics = nn.ModuleDict()   # {'head': {'train': _, 'valid': _}}
    self.is_mixed_dataset: bool = None
    self.head: str = None
    self.is_ldl: bool = None

  def setup_train_args(self):
    args = self.args
    if args.lr_list:
      lr_list: List[float] = args.lr_list
      if isinstance(lr_list, float): lr_list = [lr_list] * len(self.lr_list)
      if len(lr_list) == 1: lr_list *= len(self.lr_list)
      assert len(lr_list) == len(self.lr_list)
      self.lr_list = lr_list

    self.is_mixed_dataset = len(args.dataset) > 1
    if not self.is_mixed_dataset:
      dataset_cls = get_dataset_cls(args.dataset[0])
      self.head = dataset_cls.head.value
      self.is_ldl = dataset_cls.is_ldl

    self.metrics.clear()
    for dataset in args.dataset:
      dataset_cls = get_dataset_cls(dataset)
      head = dataset_cls.head.value
      if is_clf(head):
        self.metrics[head] = nn.ModuleDict({
          'm_train': MulticlassAccuracy(HEAD_DIMS[head]),   # avoid name conflict :(
          'm_valid': MulticlassAccuracy(HEAD_DIMS[head]),
        })
      else:
        self.metrics[head] = nn.ModuleDict({
          'm_train': MeanSquaredError(),
          'm_valid': MeanSquaredError(),
        })

  def configure_optimizers(self) -> Optimizer:
    param_groups = [
      {'params': self.model.backbone.parameters(), 'lr': self.lr_list[0]},
      {'params': self.model.proj    .parameters(), 'lr': self.lr_list[1]},
      {'params': self.model.heads   .parameters(), 'lr': self.lr_list[2]},
      {'params': self.model.invheads.parameters(), 'lr': self.lr_list[2]},
    ]
    return Adam([it for it in param_groups if it['lr'] > 0], weight_decay=1e-5)

  def forward_step(self, batch:Tuple[Tensor], batch_idx:int, prefix:str) -> Tensor:
    if self.is_mixed_dataset:
      x, y, head_id, is_ldl = batch
      head, is_ldl = HEAD_NAMES[head_id[0]], is_ldl[0].item()   # only need one
    else:
      x, y = batch
      head, is_ldl = self.head, self.is_ldl
    out, fvec, invfvec = self.model(x, head)

    if False: print(f'>> [batch {batch_idx}] head: {head}, is_clf: {is_clf(head)}, is_ldl: {is_ldl}')
    y_lbl = torch.argmax(y, dim=-1) if is_clf(head) and is_ldl else y
    if is_clf(head):
      loss_clf = F.cross_entropy(out, y_lbl)
      loss_ldl = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean', log_target=False) if is_ldl else torch.zeros_like(loss_clf)
      loss_task = loss_clf + loss_ldl * self.args.loss_w_ldl
    else:
      loss_task = F.mse_loss(out, y)
    loss_recon = F.mse_loss(invfvec, fvec)
    loss: Tensor = loss_task + loss_recon * self.args.loss_w_recon

    with torch.no_grad():
      metric = self.metrics[head][f'm_{prefix}']
      metric(out, y_lbl)
      self.log(f'{prefix}/{head}', metric, on_step=True, on_epoch=True)

      locals_kv = locals()
      log_dict = {
        'l_clf': locals_kv['loss_clf'].item() if 'loss_clf' in locals_kv else None,
        'l_ldl': locals_kv['loss_ldl'].item() if 'loss_ldl' in locals_kv else None,
        'l_task': loss_task.item(),
        'l_rec': loss_recon.item(),
        'loss': loss.item(),
      }
      log_dict = {k: v for k, v in log_dict.items() if v is not None}
      self.log_dict({f'{prefix}/{k}': v for k, v in log_dict.items()})
    return loss

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    return self.forward_step(batch, batch_idx, 'train')

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    return self.forward_step(batch, batch_idx, 'valid')

  def on_train_epoch_end(self):
    if self.is_mixed_dataset:
      trainloader: DataLoader =  self.trainer.fit_loop._data_source.instance
      mixedset: MixedDataset = trainloader.dataset
      for trainset in mixedset.datasets:
        trainset.shuffle()


def train(args):
  seed_everything(args.seed)
  print('>> cmd:', ' '.join(sys.argv))
  print('>> args:', vars(args))

  ''' Data '''
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  n_datasets = len(args.dataset)
  if n_datasets == 1:
    print('>> single dataset mode:', args.dataset[0])
    dataset_cls = get_dataset_cls(args.dataset[0])
    shuffle = True
  else:
    print('>> multi dataset mode:', args.dataset)
    dataset_cls = lambda split: MixedDataset(args.dataset, split, args.n_batch_train, args.batch_size)
    shuffle = False
  trainloader = DataLoader(dataset_cls('train'), args.batch_size, shuffle=shuffle, drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(dataset_cls('valid'), args.batch_size, shuffle=False,   drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  model = MultiTaskNet(args.model, args.head, args.d_x, pretrain=args.load is None)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model, args=args)
  else:
    lit = LitModel(model, args)
  lit.setup_train_args()

  ''' Train '''
  checkpoint_callback = ModelCheckpoint(monitor='valid/loss', mode='min')
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=n_datasets,  # 每个子数据集轮流贡献一个batch
    limit_train_batches=n_datasets*args.n_batch_train if args.n_batch_train > 0 else None,
    limit_val_batches=n_datasets*args.n_batch_valid if args.n_batch_valid > 0 else None,
    log_every_n_steps=5,
  )
  trainer.fit(lit, trainloader, validloader)


def get_parser():
  parser = ArgumentParser()
  parser.add_argument('-L', '--load',  type=Path, help='resume from pretrained weights')
  parser.add_argument('-M', '--model', default='resnet50', choices=list(BACKBONE_CLASSES.keys()), help='backbone net')
  parser.add_argument('-H', '--head',  default='linear',   choices=list(HEAD_CLASSES.keys()),     help='head net')
  parser.add_argument('-X', '--d_x', default=32, type=int, help='Xspace dim')
  parser.add_argument('-bias', '--use_bias', default=1, type=int, help='use bias in head Linear layers')
  parser.add_argument('-B', '--batch_size', type=int, default=32)
  parser.add_argument('-E', '--epochs',     type=int, default=100)
  parser.add_argument('-lr', '--lr_list', nargs='+', type=eval, default=1e-5, help='lr or lr list for each part: [backbone, proj, heads/invheads]')
  parser.add_argument('--loss_w_ldl',   type=float, default=1,  help='loss weight for ldl (kl_div loss)')
  parser.add_argument('--loss_w_recon', type=float, default=10, help='loss weight for x-space reconstruction')
  parser.add_argument('--seed', type=int, default=114514)
  return parser


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('-D', '--dataset', nargs='+', default=['EmoSet', 'Emotion6Dim7', 'Emotion6Dim6', 'Emotion6VA', 'TweeterI'], choices=DATASETS)
  parser.add_argument('--n_batch_train', default=200, type=int, help='limit n_batch for each trainset')
  parser.add_argument('--n_batch_valid', default=10,  type=int, help='limit n_batch for each validset')
  args = parser.parse_args()

  args.cmd = ' '.join(sys.argv)
  args.use_bias = bool(args.use_bias)
  train(args)
