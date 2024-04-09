@ECHO OFF

:::::::::::::::::::::::
:: Single-Task Net
:::::::::::::::::::::::
:train_baseline

REM 结论：bs大于32会过拟合，lr越小越好约1e-5量级
REM 使用同一的设置 B=32, lr=2e-5

REM 96.38%/42.76% 较平稳收敛
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 32 -lr 2e-4
REM 96.64%/42.80% 比 bs=32 更快收敛随后过拟合
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 64 -lr 2e-4
REM 97.84%/46.15% 比 bs=64 更快收敛随后过拟合
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 128 -lr 2e-4
REM 65.66%/37.38% 仍在上升，似乎lr太大
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 32 -lr 1e-3
REM 97.95%/45.72 与 lr=2e-4 基本一致且会反超，验证集收敛平稳 (*)
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 32 -lr 2e-5
REM 98.24%/41.92 后期过拟合
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 64 -lr 2e-5
REM 96.69%/45.64% 略逊于最佳设置
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 32 -lr 1e-5
REM  比 B=32 lr=4e-6 好一点，但似乎bs偏小
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 16 -lr 1e-5
REM 91.71%/47.11% 渐进 lr=2e-5 的情况，只是收敛慢
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 32 -lr 4e-6
REM 57.06%/43.57% 似乎lr太小
python train_baseline.py -M resnet50 -D Emotion6Dim6 -B 32 -lr 1e-6


:::::::::::::::::::::::
:: Multi-Task Net
:::::::::::::::::::::::

SET MODEL=resnet50
SET DATASETS=EmoSet Emotion6Dim7 Emotion6Dim6 Emotion6VA TweeterI


:train_linear
REM Step 1: refine a base ckpt with EmoSet (full weights)
python train.py -M %MODEL% -D EmoSet -lr 1e-4 --n_batch_train -1 --n_batch_valid -1

REM Step 2: train on mixed datasets (non-pretrained weights)
SET LOAD=lightning_logs\version_18\checkpoints\epoch=12-step=38376.ckpt
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 0 4e-5 1e-4

REM Step 3: refine on mixed datasets (full weights)
SET LOAD=lightning_logs\version_27\checkpoints\epoch=95-step=19200.ckpt
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 2e-6


:train_mlp
REM Step 1: refine a base ckpt with EmoSet (full weights)
python train.py -H mlp -M %MODEL% -D EmoSet -lr 1e-4 --n_batch_train -1 --n_batch_valid -1

REM Step 2: train on mixed datasets (non-pretrained weights)
SET LOAD=lightning_logs\version_33\checkpoints\epoch=14-step=44280.ckpt
python train.py -H mlp -L %LOAD% -M %MODEL% -D %DATASETS% -lr 0 4e-5 1e-4

REM Step 3: refine on mixed datasets (full weights)
SET LOAD=lightning_logs\version_36\checkpoints\epoch=99-step=20000.ckpt
python train.py -H mlp -L %LOAD% -M %MODEL% -D %DATASETS% -lr 2e-6


:visualize_xweights
python vis.py -L "lightning_logs\MTN-r50-linear\checkpoints\epoch=13-step=2800.ckpt"


:visualize_interractive
python vis_gui.py -L "lightning_logs\MTN-r50-linear\checkpoints\epoch=13-step=2800.ckpt"
python vis_gui.py -L "lightning_logs\MTN-r50-mlp\checkpoints\epoch=9-step=2000.ckpt"
