#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

import json
import random
from enum import Enum
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from utils import *

DATA_PATH = BASE_PATH / 'data'
DATA_EMOTION6_PATH = DATA_PATH / 'Emotion6'
DATA_ABSTRACT_PATH = DATA_PATH / 'testImages_abstract'
DATA_ARTPHOTO_PATH = DATA_PATH / 'testImages_artphoto'
DATA_TWEETERI_PATH = DATA_PATH / 'Twitter_PCNN'
DATA_FI_PATH = DATA_PATH / 'emotion_dataset'
DATA_GAPED_PATH = DATA_PATH / 'GAPED'
DATA_OASIS_PATH = DATA_PATH / 'OASIS_database_2016'
DATA_FER_PATH = DATA_PATH / 'FER-2013'
DATA_EMOSET_PATH = DATA_PATH / 'EmoSet-118K'

RESIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Just follow EmoSet: https://github.com/JingyuanYY/EmoSet/blob/main/EmoSet.py#L58
transform_train = T.Compose([
  T.RandomResizedCrop(RESIZE),
  T.RandomHorizontalFlip(),
  T.ToTensor(),
  T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
transform_test = T.Compose([
  T.Resize(RESIZE),
  T.CenterCrop(RESIZE),
  T.ToTensor(),
  T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

LABELS_MIKELS  = ['anger', 'disgust', 'fear', 'amusement', 'sadness', 'excitement', 'contentment', 'awe']
LABELS_EKMAN   = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
LABELS_EKMAN_N = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
LABELS_VA      = ['valence', 'arousal']
LABELS_POLAR   = ['neg', 'pos']

class HeadType(Enum):

  Mikels = 'Mikels'
  Ekman = 'Ekman'
  EkmanN = 'EkmanN'
  VA = 'VA'
  Polar = 'Polar'

HEAD_CLASS_NAMES = {
  # 'head_type': dim
  'Mikels': LABELS_MIKELS,   # 8
  'EkmanN': LABELS_EKMAN_N,  # 7
  'Ekman':  LABELS_EKMAN,    # 6
  'VA':     LABELS_VA,       # 2
  'Polar':  LABELS_POLAR,    # 2
}
HEAD_DIMS = { k: len(v) for k, v in HEAD_CLASS_NAMES.items() }
HEAD_NAMES = list(HEAD_DIMS.keys())

is_clf = lambda head: head not in [HeadType.VA.value]


def split_dataset(metadata:list, split:str='train', split_ratio:float=0.2) -> Tuple[list, list]:
  import random
  random.seed(114514)
  random.shuffle(metadata)
  cp = int(len(metadata) * split_ratio)
  return metadata[cp:] if split == 'train' else metadata[:cp]


''' Base '''

class BaseDataset(Dataset):

  root: Path = None
  head: HeadType = None
  is_ldl: bool = False

  def __init__(self, split:str='train'):
    assert split in ['train', 'valid', 'test']
    self.split = split
    self.img_root = self.root
    self.transform = transform_train if split == 'train' else transform_test
    self._metadata = []

  @property
  def class_names(self):
    return HEAD_CLASS_NAMES[self.head.value]

  @property
  def metadata(self):
    return self._metadata

  def get_fps(self):
    return [(self.img_root / mt[0]) for mt in self.metadata]

  def __len__(self):
    return len(self.metadata)

  def shuffle(self):
    random.shuffle(self._metadata)


''' EmoSet '''

class EmoSet(BaseDataset):

  root = DATA_EMOSET_PATH
  head = HeadType.Mikels

  def __init__(self, split:str='train'):
    super().__init__(split)

    if self.split == 'valid': self.split = 'val'
    assert self.split in ['train', 'val', 'test']
    with open(self.root / f'{self.split}.json', 'r', encoding='utf-8') as fh:
      self._metadata = json.load(fh)

  def get_fps(self):
    return [(self.root / mt[1]) for mt in self.metadata]

  def __getitem__(self, idx:int):
    lbl, rfp, annot = self.metadata[idx]
    img = load_pil(self.root / rfp)
    img = self.transform(img)
    return img, self.class_names.index(lbl)


''' Emotion6 '''

class Emotion6BaseDataset(BaseDataset):

  root = DATA_EMOTION6_PATH

  def __init__(self, split:str='train', split_ratio:float=0.2):
    super().__init__(split)

    self.img_root = self.root / 'images'
    df = pd.read_csv(self.root / 'ground_truth.txt', sep='\t').to_numpy()
    X, Y = df[:, 0], df[:, 1:]
    self._metadata = split_dataset([(x, y) for x, y in zip(X, Y)], split, split_ratio)

  def get_img(self, idx:int) -> Tensor:
    fn, _ = self.metadata[idx]
    img = load_pil(self.root / 'images' / fn)
    img = self.transform(img)
    return img

class Emotion6Dim7(Emotion6BaseDataset):

  head = HeadType.EkmanN
  is_ldl = True

  def __getitem__(self, idx:int):
    img = self.get_img(idx)
    _, vals = self.metadata[idx]
    prob = np.asarray(vals[2:], dtype=np.float32)
    return img, prob

class Emotion6Dim6(Emotion6BaseDataset):

  head = HeadType.Ekman
  is_ldl = True

  def __getitem__(self, idx:int):
    img = self.get_img(idx)
    _, vals = self.metadata[idx]
    prob = np.asarray(vals[2:8], dtype=np.float32)
    prob /= prob.sum()    # re-norm to 1
    return img, prob

class Emotion6VA(Emotion6BaseDataset):

  head = HeadType.VA

  def __getitem__(self, idx:int):
    img = self.get_img(idx)
    _, vals = self.metadata[idx]
    vals = np.asarray(vals[:2], dtype=np.float32)
    return img, vals


''' Abstract & ArtPhoto '''

class Abstract(BaseDataset):

  root = DATA_ABSTRACT_PATH
  head = HeadType.Mikels
  is_ldl = True

  def __init__(self, split:str='train', split_ratio:float=0.2):
    super().__init__(split)

    original_class_names = ['Amusement', 'Anger', 'Awe', 'Content', 'Disgust', 'Excitement', 'Fear', 'Sad']
    tmp_class_names = [e.lower().replace('content', 'contentment').replace('sad', 'sadness') for e in original_class_names]

    df = pd.read_csv(self.root / 'ABSTRACT_groundTruth.csv').to_numpy()
    X, Y = df[:, 0], df[:, 1:]
    X = [fn.strip("'") for fn in X]
    Y = Y[:, [tmp_class_names.index(e) for e in self.class_names]]  # re-order for mapping
    Y /= Y.sum(axis=-1, keepdims=True)  # freq to prob
    self._metadata = split_dataset([(x, y) for x, y in zip(X, Y)], split, split_ratio)

  def __getitem__(self, idx:int) -> int:
    fn, prob = self.metadata[idx]
    img = load_pil(self.root / fn)
    img = self.transform(img)
    return img, prob

class ArtPhoto(BaseDataset):

  root = DATA_ARTPHOTO_PATH
  head = HeadType.Mikels

  def __init__(self, split:str='train', split_ratio:float=0.2):
    super().__init__(split)

    X = [fp.name for fp in self.root.iterdir() if fp.suffix == '.jpg']
    Y = [self.class_names.index(x.split('_')[0].replace('sad', 'sadness')) for x in X]
    self._metadata = split_dataset([(x, y) for x, y in zip(X, Y)], split, split_ratio)

  def __getitem__(self, idx:int) -> int:
    fn, lbl = self.metadata[idx]
    img = load_pil(self.root / fn)
    img = self.transform(img)
    return img, lbl


''' Tweeter-I & FI-23k (Flickr-Instagram) '''

class TweeterI(BaseDataset):

  root = DATA_TWEETERI_PATH
  head = HeadType.Polar
  is_ldl = True

  def __init__(self, split:str='train', split_ratio:float=0.2):
    super().__init__(split)

    self.img_root = self.root / 'Agg_AMT_Candidates'
    df = pd.read_csv(self.root / 'amt_result.csv').to_numpy()
    X = df[:, 0]
    Y = np.stack([df[:, 2] / df[:, 1], df[:, 3] / df[:, 1]], axis=-1).astype(np.float32)
    self._metadata = split_dataset([(x, y) for x, y in zip(X, Y)], split, split_ratio)

  def __getitem__(self, idx:int) -> int:
    fn, prob = self.metadata[idx]
    img = load_pil(self.root / 'Agg_AMT_Candidates' / fn)
    img = self.transform(img)
    return img, prob

class FI(BaseDataset):

  pass


''' others '''

class GAPED(BaseDataset):

  root = DATA_GAPED_PATH

class OASIS(BaseDataset):

  root = DATA_OASIS_PATH

class FER2013(BaseDataset):
  
  root = DATA_FER_PATH


DATASETS = [k for k, v in globals().items() if type(v) == type(BaseDataset) and issubclass(v, BaseDataset) and v not in [BaseDataset, Emotion6BaseDataset]]

def get_dataset_cls(name:str) -> 'BaseDataset':
  assert name in DATASETS
  return globals()[name]


if __name__ == '__main__':
  keys = list(globals().keys())
  for k in keys:
    v = globals()[k]
    if v in [BaseDataset, Emotion6BaseDataset]: continue
    if not type(v) == type(BaseDataset): continue
    if not issubclass(v, BaseDataset): continue

    try:
      dataset: BaseDataset = v('train')
      print(f'>> [{k}] len={len(dataset)} ({dataset.head.value})')
      for x, y in iter(dataset):
        print(f'  x.shape: {tuple(x.shape)}')
        if isinstance(y, int):
          print(f'  y: {y}')
        elif isinstance(y, ndarray):
          print(f'  y.shape: {tuple(y.shape)}' if isinstance(y, ndarray) else f'  y: {y}')
          print(f'  y: {[round(e, 4) for e in y]}, sum: {sum(y):.4f}')
        else: raise TypeError(type(y))
        break
      print()
    except:
      #from traceback import print_exc
      #print_exc()
      print(f'>> [{k}] failed!')
