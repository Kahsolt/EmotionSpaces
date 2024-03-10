#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/10

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, root_mean_squared_error, r2_score

import numpy as np
from numpy import ndarray

EPS = 1e-15


''' classification '''

def acc(y_hat:ndarray, y_truth:ndarray) -> float:
  return accuracy_score(y_truth, y_hat)

def prec(y_hat:ndarray, y_truth:ndarray, average:str='weighted') -> float:
  return precision_score(y_truth, y_hat, average=average)

def recall(y_hat:ndarray, y_truth:ndarray, average:str='weighted') -> float:
  return recall_score(y_truth, y_hat, average=average)

def f1(y_hat:ndarray, y_truth:ndarray, average:str='weighted') -> float:
  return f1_score(y_truth, y_hat, average=average)


''' regression '''

def mae(y_hat:ndarray, y_truth:ndarray) -> float:
  return mean_absolute_error(y_truth, y_hat)

def mse(y_hat:ndarray, y_truth:ndarray) -> float:
  return mean_squared_error(y_truth, y_hat)

def msle(y_hat:ndarray, y_truth:ndarray) -> float:
  return mean_squared_log_error(y_truth, y_hat)

def rmse(y_hat:ndarray, y_truth:ndarray) -> float:
  return root_mean_squared_error(y_truth, y_hat)

def r2(y_hat:ndarray, y_truth:ndarray) -> float:
  return r2_score(y_truth, y_hat)


''' label distribution learning '''

# ref: https://arxiv.org/pdf/1408.6027.pdf

def vectorize(fn) -> float:
  def wrapper(y_hat:ndarray, y_truth:ndarray):
    assert y_hat.shape == y_truth.shape
    return np.mean([fn(a, b) for a, b in zip(y_truth, y_hat)]).item()
  return wrapper

@vectorize
def chebyshev_dist(x:ndarray, y:ndarray) -> float:
  return np.max(np.abs(y - x))

@vectorize
def clark_dist(x:ndarray, y:ndarray) -> float:
  return np.sqrt(np.sum(np.square(y - x) / np.square(y + x)))

@vectorize
def canberra_dist(x:ndarray, y:ndarray) -> float:
  return np.sum(np.abs(y - x) / (y + x))

@vectorize
def kullback_leibler_dist(x:ndarray, y:ndarray) -> float:
  return np.sum(y * (np.log(y + EPS) - np.log(x + EPS)))

@vectorize
def cosine_sim(x:ndarray, y:ndarray) -> float:
  return np.sum(y * x) / (np.sqrt(np.sum(np.square(y))) * np.sqrt(np.sum(np.square(x))) + EPS)

@vectorize
def intersection_sim(x:ndarray, y:ndarray) -> float:
  return np.sum(np.where(y < x, y, x))


if __name__ == '__main__':
  CLF_METRICS = [
    acc,
    prec,
    recall,
    f1,
  ]
  y = np.asarray([0, 1, 2, 3, 3])   # truth
  x = np.asarray([3, 1, 2, 0, 2])   # pred
  print('[clf]')
  for metric in CLF_METRICS:
    print(metric(x, y))

  print()

  RGR_METRICS = [
    mae,
    mse,
    msle,
    rmse,
    r2,
  ]
  y = np.asarray([[0.1, 0.6], [0.7, 0.5]])   # truth
  x = np.asarray([[0.1, 0.5], [0.8, 0.5]])   # pred
  print('[rgr]')
  for metric in RGR_METRICS:
    print(metric(x, y))

  print()

  LDL_METRICS = [
    chebyshev_dist,
    clark_dist,
    canberra_dist,
    kullback_leibler_dist,
    cosine_sim,
    intersection_sim,
  ]
  y = np.asarray([[0.1, 0.6, 0.0, 0.3]])   # truth
  x = np.asarray([[0.2, 0.4, 0.1, 0.3]])   # pred
  print('[ldl]')
  for metric in LDL_METRICS:
    print(metric(x, y))
