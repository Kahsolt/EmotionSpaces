#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/15

from data import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import imagesize

log_dp = IMG_PATH / 'hw'
log_dp.mkdir(exist_ok=True)

DATASETS_IGNORE = [
  'Emotion6Dim7',
  'Emotion6VA',
]

for name in DATASETS:
  if name in DATASETS_IGNORE: continue
  save_fp = log_dp / f'{name}.png'
  if save_fp.exists():
    print(f'>> ignore {name} due to file exists')
    continue

  print(f'>> process {name}')
  dataset_cls = get_dataset_cls(name)

  try:
    plt.clf()
    plt.figure(figsize=(8, 4))
    for idx, (split, cmap) in enumerate(zip(['train', 'valid'], ['blue', 'red'])):
      hs, ws = [], []
      dataset: BaseDataset = dataset_cls(split)
      for fp in tqdm(dataset.get_fps()):
        w, h = imagesize.get(fp)
        hs.append(h)
        ws.append(w)
      plt.subplot(120 + idx + 1)
      plt.scatter(ws, hs, c=cmap, alpha=0.5, label=split)
      plt.xlabel('width')
      plt.ylabel('height')

    if not hs: continue
    plt.tight_layout()
    plt.savefig(save_fp)
    plt.close()
  except KeyboardInterrupt:
    exit(-1)
  except Exception as e:
    print(e)
    print(f'>> {name} failed')
