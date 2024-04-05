### Challenge

- 现存模型过拟合
- VA这种连续空间难以理解


### Model

> hparam: E=100, B=32, lr=2e-5

#### baseline

| Accuracy (train/valid) | ResNet50 | ResNet101 | MobileNet_V2 | ViT_B_16 | ViT_B_32 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Emo6Dim6 | 97.95%/45.72% |  |  |  |  |  |
| Emo6Dim7 |  |  |  |  |  |  |
| Emo6VA   |  |  |  |  |  |  |
| Abstract |  |  |  |  |  |  |
| Artphoto |  |  |  |  |  |  |
| OASIS    |  |  |  |  |  |  |
| TwitterI |  |  |  |  |  |  |
| FI       |  |  |  |  |  |  |
| EmoSet   |  |  |  |  |  |  |

> 别人论文或者项目代码里抄的

| Accuracy | xxx | xxx | xxx | xxx | xxx |
| :-: | :-: | :-: | :-: | :-: | :-: |
| xxx |  |  |  |  |  |  |

#### ours

head_type: Polar, VA. Ekman, EkmanN, Mikels
head_name: 9 heads

⚪ Head=linear (数值精度可能低一些，但可以看出相关性)

> run vis.py => img/tx/*.png 可视化权重矩阵/相关性
> run vis_gui.py => 交互式可视化

| Accuracy | M-ResNet50 | M-MobileNet_V2 | M-ViT_B_16 |
| :-: | :-: | :-: | :-: |
| Emo6Dim6 | 97.95%/45.72% |  |  |  |
| Emo6Dim7 |  |  |  |  |
| Emo6VA   |  |  |  |  |
| Abstract |  |  |  |  |
| Artphoto |  |  |  |  |
| OASIS    |  |  |  |  |
| TwitterI |  |  |  |  |
| FI       |  |  |  |  |
| EmoSet   |  |  |  |  |

⚪ Head=mlp (数值精度可能高一些，但无法透视相关性)

> run vis_gui.py => 交互式可视化

| Accuracy | M-ResNet50 | M-MobileNet_V2 | M-ViT_B_16 |
| :-: | :-: | :-: | :-: |
| Emo6Dim6 | 97.95%/45.72% |  |  |  |
| Emo6Dim7 |  |  |  |  |
| Emo6VA   |  |  |  |  |
| Abstract |  |  |  |  |
| Artphoto |  |  |  |  |
| OASIS    |  |  |  |  |
| TwitterI |  |  |  |  |
| FI       |  |  |  |  |
| EmoSet   |  |  |  |  |


### Analysis

- 多任务模型 (9 datasets, 3/5 model archs, 2 head types)
  - 精度是否更好，最好略好
  - 有没有减缓过拟合
  - 在精度不下降太多的情况下，一定程度上节省了参数
  - LDL六种度量
- 空间转换
  - 可视化权重矩阵/相关性: `run vis.py => img/tx/*.png`
  - 交互式可视化: `run vis_gui.py`
