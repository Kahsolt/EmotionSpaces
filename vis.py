#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/27

# 查看各空间之间的投影变换

from train import *

import yaml
import seaborn as sns
import matplotlib.pyplot as plt

device = 'cpu'

EX_BIAS = ['(bias)']


def get_w_and_b(head:LinearHead) -> Tuple[ndarray, ndarray]:
  assert isinstance(head, LinearHead), f'>> only support LinearHead, but got: {type(head)}'
  layer = head.fc
  w = layer.weight.detach().cpu().numpy()   # [d_out, d_in]
  b = layer.bias  .detach().cpu().numpy()   # [d_out]
  return w, b

def expanded_matrix(w:ndarray, b:ndarray) -> ndarray:
  assert len(w.shape) == 2
  assert len(b.shape) == 1
  assert w.shape[0] == b.shape[0]
  return np.concatenate([w, np.expand_dims(b, axis=-1)], axis=-1)

def seqnum_label(nlen:int) -> List[str]:
  return [str(e + 1) for e in range(nlen)]


def savefig(mat:ndarray, xticks:Tuple[List[str]], yticks:Tuple[List[str]], title:str, figsize:Tuple[int, int], fp:Path):
  H, W = mat.shape
  if H > W:
    mat = mat.T
    xticks, yticks = yticks, xticks

  plt.clf()
  plt.figure(figsize=figsize)
  sns.heatmap(mat.T, annot=True, cbar=True)
  plt.gca().invert_yaxis()
  plt.xticks(*xticks, rotation=0, fontsize=8)
  plt.yticks(*yticks, rotation=0, fontsize=8)
  plt.suptitle(title)
  plt.tight_layout()
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=600)
  plt.close()


def vis_tx_x2h(model:MultiTaskNet, out_dp:Path):
  ''' 从 交换空间(X-space) 到 各head 的出入投影转换 '''

  figsize = (8, 8)
  for name in list(model.heads.keys()):
    try:
      w1, b1 = get_w_and_b(model.heads[name])
      w2, b2 = get_w_and_b(model.invheads[name])
      D, X = w1.shape
      mat_ex1 = expanded_matrix(w1, b1)   # [D, X+1]
      mat_ex2 = expanded_matrix(w2, b2)   # [X, D+1]

      xticks = 0.5 + np.arange(X+1), seqnum_label(X) + EX_BIAS
      yticks = 0.5 + np.arange(D), HEAD_CLASS_NAMES[name]
      title = f'X-space -> {name}'
      fp = out_dp / f'Xspace-{name}.png'
      savefig(mat_ex1.T, xticks, yticks, title, figsize, fp)

      xticks = 0.5 + np.arange(D+1), HEAD_CLASS_NAMES[name] + EX_BIAS
      yticks = 0.5 + np.arange(X), seqnum_label(X)
      title = f'{name} -> X-space'
      fp = out_dp / f'{name}-Xspace.png'
      savefig(mat_ex2.T, xticks, yticks, title, figsize, fp)
    except KeyboardInterrupt:
      exit(-1)
    except:
      print(f'>> [vis_tx_x2h] failed: {name}')


def vis_tx_h2h(model:MultiTaskNet, out_dp:Path):
  ''' 从 一个head 到 另一个head 的投影转换 '''

  figsize = (6, 6)
  for src in list(model.heads.keys()):
    for dst in list(model.heads.keys()):
      try:
        # dst = head(invhead(src))
        w1, b1 = get_w_and_b(model.invheads[src])
        w2, b2 = get_w_and_b(model.heads[dst])
        # Y = w2 * (w1 * X + b1) + b2
        w_tx = w2 @ w1
        b_tx = w2 @ b1 + b2
        mat_ex = expanded_matrix(w_tx, b_tx)  # [D+1, D']

        xticks = 0.5 + np.arange(mat_ex.shape[1]), HEAD_CLASS_NAMES[src] + EX_BIAS
        yticks = 0.5 + np.arange(mat_ex.shape[0]), HEAD_CLASS_NAMES[dst]
        title = f'{src} -> {dst}'
        fp = out_dp / f'{src}-{dst}.png'
        savefig(mat_ex.T, xticks, yticks, title, figsize, fp)
      except KeyboardInterrupt:
        exit(-1)
      except:
        print(f'>> [vis_tx_h2h] failed: {src} => {dst}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--load', type=Path, required=True, help='load pretrained weights')
  args = parser.parse_args()

  ''' Model & Ckpt '''
  fp = Path(args.load).parent.parent / 'hparams.yaml'
  with open(fp, 'r', encoding='utf-8') as fh:
    hp = yaml.safe_load(fh)
  model = MultiTaskNet(hp['model'], hp['head'], hp['d_x'], pretrain=False)
  model = LitModel.load_from_checkpoint(args.load, model=model).model.to(device).eval()

  IMG_PATH.mkdir(exist_ok=True)
  out_dp = IMG_PATH / 'tx' ; out_dp.mkdir(exist_ok=True)
  vis_tx_x2h(model, out_dp)
  vis_tx_h2h(model, out_dp)
