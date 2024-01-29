#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/27

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from utils import *
from utils import Path

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


''' Emotion6 '''

class Emotion6(Dataset):

  transform_train = T.Compose([
    T.RandomResizedCrop(RESIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
  ])
  transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

  def __init__(self, split:str='train', root:Path=DATA_EMOTION6_PATH):
    self.root = root
    self.split = split
    self.transform = self.transform_train if split == 'train' else self.transform_test
    self.metadata = pd.read_csv(root / 'ground_truth.txt', sep='\t')

  @property
  def n_class(self):
    return len(self.class_names)

  def __len__(self):
    return len(self.metadata)

  def meta(self, idx:int):
    return self.metadata.iloc[idx].tolist()

  def get_img(self, idx:int) -> Tensor:
    row = self.meta(idx)
    img = load_pil(self.root / 'images' / row[0])
    img = self.transform(img)
    return img

class Emotion6Dim7(Emotion6):

  class_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

  def __getitem__(self, idx:int):
    img = self.get_img(idx)
    row = self.meta(idx)
    prob = np.asarray(row[3:9], dtype=np.float32)
    return img, prob

class Emotion6Dim6(Emotion6):

  class_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

  def __getitem__(self, idx:int):
    img = self.get_img(idx)
    row = self.meta(idx)
    prob = np.asarray(row[3:8], dtype=np.float32)
    prob /= prob.sum()    # re-norm to 1
    return img, prob

class Emotion6VA(Emotion6):

  class_names = ['valence', 'arouse']

  def __getitem__(self, idx:int):
    img = self.get_img(idx)
    row = self.meta(idx)
    prob = np.asarray(row[1:3], dtype=np.float32)
    return img, prob


''' Abstract & ArtPhoto '''

class Abstract(Dataset):

  pass

class ArtPhoto(Dataset):

  pass


''' Tweeter-I & FI-23k (Flickr-Instagram) '''

class TweeterI(Dataset):
  
  pass

class FI(Dataset):
  
  pass


''' others '''

class GAPED(Dataset):
  
  pass

class OASIS(Dataset):
  
  pass

class FER2013(Dataset):
  
  pass

class EmoSet(Dataset):
  
  pass


if __name__ == '__main__':
  dataset = Emotion6Dim6()
  for img, label in iter(dataset):
    print('x:', img, 'y:', label)
    print(np.sum(label))
