#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/08

from train import *

is_clf = lambda head: head not in [HeadType.VA.value]


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


class MixedLitModel(LightningModule):

  def __init__(self, model:MultiTaskResNet):
    super().__init__()

    self.model = model
    # ↓↓ training specified ↓↓
    self.args = None
    self.lr_list: List[float] = [1e-6, 1e-4, 2e-4]
    self.metrics = nn.ModuleDict()   # {'head': {'train': _, 'valid': _}}

  def setup_train_args(self, args):
    self.args = args
    if args.lr_list:
      lr_list: List[float] = args.lr_list
      if isinstance(lr_list, float): lr_list = [lr_list] * len(self.lr_list)
      if len(lr_list) == 1: lr_list *= len(self.lr_list)
      assert len(lr_list) == len(self.lr_list)
      self.lr_list = lr_list

    self.metrics.clear()
    for dataset in args.dataset:
      head = DATASET_TO_HEAD_TYPE[dataset]
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
      {'params': self.model.fvecs   .parameters(), 'lr': self.lr_list[0]},
      {'params': self.model.proj    .parameters(), 'lr': self.lr_list[1]},
      {'params': self.model.heads   .parameters(), 'lr': self.lr_list[2]},
      {'params': self.model.invheads.parameters(), 'lr': self.lr_list[2]},
    ]
    return Adam([it for it in param_groups if it['lr'] > 0], weight_decay=1e-5)

  def forward_step(self, batch:Tuple[Tensor], batch_idx:int, prefix:str) -> Tensor:
    x, y, head_id, is_ldl = batch
    head, is_ldl = HEAD_NAMES[head_id[0]], is_ldl[0].item()   # only need one
    out, fvec, invfvec = self.model(x, head)

    if False: print(f'>> [batch {batch_idx}] head: {head}, is_clf: {is_clf(head)}, is_ldl: {is_ldl}')
    metric = self.metrics[head][f'm_{prefix}']
    y_lbl = torch.argmax(y, dim=-1) if is_clf(head) and is_ldl else y
    if is_clf(head):
      loss_clf = F.cross_entropy(out, y_lbl)
      loss_ldl = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean') if is_ldl else 0.0
      loss_task = loss_clf + loss_ldl * self.args.loss_w_ldl
    else:
      loss_task = F.mse_loss(out, y)
    metric(out, y_lbl)
    self.log(f'{prefix}/{head}', metric, on_step=prefix=='train', on_epoch=True)

    loss_recon = F.mse_loss(invfvec, fvec)
    loss: Tensor = loss_task + loss_recon * self.args.loss_w_recon

    with torch.no_grad():
      log_dict = {
        'l_clf': locals().get('loss_clf').item() if locals().get('loss_clf') else None,
        'l_ldl': locals().get('loss_ldl').item() if locals().get('loss_ldl') else None,
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
  trainloader = DataLoader(MixedDataset(args.dataset, 'train', args.n_batch_train, args.batch_size), args.batch_size, shuffle=False, drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(MixedDataset(args.dataset, 'valid', args.n_batch_valid, batch_size=1   ), batch_size=1,    shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  model = MultiTaskResNet(args.model, args.d_x, args.head, pretrain=args.load is None)
  lit = MixedLitModel(model)
  if args.load:
    lit = MixedLitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args)

  ''' Train '''
  n_datasets = len(args.dataset)
  checkpoint_callback = ModelCheckpoint(monitor='valid/loss', mode='min')
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=n_datasets,  # 每个子数据集轮流贡献一个batch
    limit_train_batches=n_datasets*args.n_batch_train,
    limit_val_batches=n_datasets*args.n_batch_valid,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('-D', '--dataset', nargs='+', default=['EmoSet', 'Emotion6Dim7', 'Emotion6Dim6', 'Emotion6VA', 'TweeterI'], choices=DATASETS)
  parser.add_argument('--n_batch_train', default=200, type=int, help='limit n_batch for each trainset')
  parser.add_argument('--n_batch_valid', default=100, type=int, help='limit n_batch for each validset')
  args = parser.parse_args()

  train(args)
