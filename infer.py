#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/29

from train_utils import *

CKPT_FILE = './lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt'


@torch.inference_mode()
def infer():
  model = MultiTaskNet()
  lit = BaseLitModel.load_from_checkpoint(CKPT_FILE, model=model)
  model = lit.model.eval()
  x = torch.rand([1, 3, 224, 224])
  pred = model(x, 'VA')
  print(pred)


if __name__ == '__main__':
  infer()
