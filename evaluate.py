#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/27

# 0. 训练一个模型，保存好权重
# 1. 加载预训练好的权重，恢复现场，布置推理环境
#   - 预训练权重
#   - 测试数据 (测训练时用的所有数据集)
# 2. 过一遍dataloader，计算每个batch的性能指标，记录下来
# 3. 统计整个数据集上的该性能指标

import yaml
from tqdm import tqdm

from train import *
from train import is_clf as is_clf_fn
from metrics import *


@torch.inference_mode()
def run(args):
  ''' Model & Ckpt '''
  fp = Path(args.load).parent.parent / 'hparams.yaml'
  with open(fp, 'r', encoding='utf-8') as fh:
    hp = yaml.unsafe_load(fh)
  model = MultiTaskNet(hp['model'], hp['head'], hp['d_x'], pretrain=False)
  model = LitModel.load_from_checkpoint(args.load, model=model).model.to(device).eval()

  ''' Data '''
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': False,
  }
  for dataset in (args.dataset or hp['dataset']):
    dataset_cls = get_dataset_cls(dataset)
    dataloader = DataLoader(dataset_cls(args.split), args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

    ''' Bookkeep '''
    head = dataset_cls.head.value
    is_ldl = dataset_cls.is_ldl
    is_clf = is_clf_fn(dataset_cls.head.value)

    ''' Evaluate '''
    Y_clf_list: List[Tensor] = []
    Y_ldl_list: List[Tensor] = []
    Y_rgr_list: List[Tensor] = []
    pred_list: List[Tensor] = []
    prob_list: List[Tensor] = []
    output_list: List[Tensor] = []
    for X, Y in tqdm(dataloader):
      X, Y = X.to(device), Y.to(device)

      if is_clf:
        Y_clf = torch.argmax(Y, dim=-1) if is_ldl else Y
        Y_clf_list.append(Y_clf)
        if is_ldl:
          Y_ldl_list.append(Y)
      else:
        Y_rgr_list.append(Y)

      out, _, _ = model(X, head)

      if is_clf:
        pred_list.append(torch.argmax(out, dim=-1))
        if is_ldl:
          prob_list.append(F.softmax(out, dim=-1))
      else:
        output_list.append(out)


    print(f'>> [{dataset}]')

    if is_clf:
      preds  = torch.cat(pred_list ).cpu().numpy()
      Y_clfs = torch.cat(Y_clf_list).cpu().numpy()

      acc_v    = acc   (preds, Y_clfs)
      prec_v   = prec  (preds, Y_clfs)
      recall_v = recall(preds, Y_clfs)
      f1_v     = f1    (preds, Y_clfs)

      print(f'>> acc: {acc_v:.3%}')
      print(f'>> prec: {prec_v:.3%}')
      print(f'>> recall: {recall_v:.3%}')
      print(f'>> f1: {f1_v:.3%}')

      if is_ldl:
        probs  = torch.cat(prob_list ).cpu().numpy()
        Y_ldls = torch.cat(Y_ldl_list).cpu().numpy()

        chebyshev_v        = chebyshev_dist       (probs, Y_ldls)
        clark_v            = clark_dist           (probs, Y_ldls)
        canberra_v         = canberra_dist        (probs, Y_ldls)
        kullback_leibler_v = kullback_leibler_dist(probs, Y_ldls)
        cosine_sim_v       = cosine_sim           (probs, Y_ldls)
        intersection_sim_v = intersection_sim     (probs, Y_ldls)

        print(f'>> chebyshev: {chebyshev_v:.5f}')
        print(f'>> clark: {clark_v:.5f}')
        print(f'>> canberra: {canberra_v:.5f}')
        print(f'>> kl_div: {kullback_leibler_v:.5f}')
        print(f'>> cos_sim: {cosine_sim_v:.5f}')
        print(f'>> intersect_sim: {intersection_sim_v:.5f}')

    else:
      outputs = torch.cat(output_list).cpu().numpy()
      Y_rgrs  = torch.cat(Y_rgr_list ).cpu().numpy()

      mae_v  = mae (outputs, Y_rgrs)
      mse_v  = mse (outputs, Y_rgrs)
      #msle_v = msle(outputs, Y_rgrs)
      rmse_v = rmse(outputs, Y_rgrs)
      r2_v   = r2  (outputs, Y_rgrs)

      print(f'>> mae: {mae_v:.5f}')
      print(f'>> mse: {mse_v:.5f}')
      #print(f'>> msle: {msle_v:.5f}')
      print(f'>> rmse: {rmse_v:.5f}')
      print(f'>> r2: {r2_v:.5f}')
      

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--load', type=Path, required=True, help='load pretrained weights')
  parser.add_argument('-D', '--dataset', default=[], nargs='+', help='override dataset for evaluate')
  parser.add_argument('--split', default='valid', choices=['train', 'valid'], help='dataset split')
  parser.add_argument('-B', '--batch_size', type=int, default=128)
  args = parser.parse_args()

  run(args)
