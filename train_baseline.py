#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/11

# 训练单任务模型 (基线对比)

from train import *

SingleTaskNet = Union[ResNet, VisionTransformer, MobileNetV2]


def fix_last_layer(model:SingleTaskNet, n_class:int):
  if isinstance(model, ResNet):
    layer = model.fc
    model.fc = nn.Linear(layer.in_features, n_class, exists(layer.bias))
  elif isinstance(model, VisionTransformer):
    layer = model.heads[-1]
    model.heads[-1] = nn.Linear(layer.in_features, n_class, exists(layer.bias))
  elif isinstance(model, MobileNetV2):
    layer: nn.Linear = model.classifier[-1]
    model.classifier[-1] = nn.Linear(layer.in_features, n_class, exists(layer.bias))


class LitModel(LightningModule):

  def __init__(self, model:SingleTaskNet, args:Namespace=None):
    super().__init__()

    assert isinstance(model, SingleTaskNet), f'>> model must be SingleTaskNet type, but got: {type(model)}'
    self.model = model

    # ↓↓ training specified ↓↓
    if args: self.save_hyperparameters(args)
    self.args = args
    self.head: str = None
    self.is_ldl: bool = None
    self.metrics = nn.ModuleDict()   # {'train': _, 'valid': _}

  def setup_train_args(self):
    args = self.args
    assert isinstance(self.args.lr, float), '>> --lr must be single float number'
    dataset_cls = get_dataset_cls(args.dataset)
    self.head = dataset_cls.head.value
    self.is_ldl = dataset_cls.is_ldl
    self.is_clf = is_clf(self.head)
    self.metrics['m_train'] = MulticlassAccuracy(HEAD_DIMS[self.head]) if self.is_clf else MeanSquaredError()
    self.metrics['m_valid'] = MulticlassAccuracy(HEAD_DIMS[self.head]) if self.is_clf else MeanSquaredError()

  def configure_optimizers(self) -> Optimizer:
    return Adam(self.model.parameters(), self.args.lr, weight_decay=1e-5)

  def forward_step(self, batch:Tuple[Tensor], batch_idx:int, prefix:str) -> Tensor:
    x, y = batch
    out = self.model(x)

    y_lbl = torch.argmax(y, dim=-1) if self.is_ldl else y
    if self.is_clf:
      loss_clf = F.cross_entropy(out, y_lbl)
      loss_ldl = F.kl_div(F.log_softmax(out, dim=-1), y, reduction='batchmean', log_target=False) if self.is_ldl else 0.0
      loss = loss_clf + loss_ldl * self.args.loss_w_ldl
    else:
      loss = F.mse_loss(out, y)

    with torch.no_grad():
      metric = self.metrics[f'm_{prefix}']
      metric(out, y_lbl)
      self.log(f'{prefix}/{self.head}', metric, on_step=True, on_epoch=True)
      
      locals_kv = locals()
      log_dict = {
        'l_clf': locals_kv['loss_clf'].item() if 'loss_clf' in locals_kv else None,
        'l_ldl': locals_kv['loss_ldl'].item() if 'loss_ldl' in locals_kv else None,
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
  dataset_cls: BaseDataset = get_dataset_cls(args.dataset)
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainloader = DataLoader(dataset_cls('train'), args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(dataset_cls('valid'), args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  model = globals()[args.model](pretrained=args.load is None)
  fix_last_layer(model, HEAD_DIMS[dataset_cls.head.value])
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
    limit_train_batches=pos_or_none(args.n_batch_train),
    limit_val_batches=pos_or_none(args.n_batch_valid),
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('-D', '--dataset', default='Emotion6Dim6', choices=DATASETS)
  parser.add_argument('--n_batch_train', default=-1, type=int, help='limit n_batch for trainset')
  parser.add_argument('--n_batch_valid', default=-1, type=int, help='limit n_batch for validset')
  args = parser.parse_args()

  args.cmd = ' '.join(sys.argv)
  args.lr = args.lr_list[0]
  train(args)
