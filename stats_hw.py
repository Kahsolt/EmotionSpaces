#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/15

from data import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import imagesize

log_dp = IMG_PATH / 'hw'
log_dp.mkdir(exist_ok=True)

DATASETS = [
  'Emotion6Dim6',
  'Abstract',
  'ArtPhoto',
  'TweeterI',
  'EmoSet',
]

for name in DATASETS:
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
    plt.savefig(log_dp / f'{name}.png')
    plt.close()
  except KeyboardInterrupt:
    exit(-1)
  except Exception as e:
    print(e)
    print(f'>> {name} failed')
