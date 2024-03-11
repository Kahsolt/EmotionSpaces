@ECHO OFF

REM 结论：bs大于32会过拟合，lr越小越好约1e-5量级

REM 96.38%/42.76% 较平稳收敛 (*)
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


python train_baseline.py -M resnet50 -D Emotion6VA -B 32 -lr 1e-3
python train_baseline.py -M resnet50 -D Emotion6VA -B 32 -lr 2e-4
python train_baseline.py -M resnet50 -D Emotion6VA -B 64 -lr 2e-4
python train_baseline.py -M resnet50 -D Emotion6VA -B 128 -lr 2e-4
python train_baseline.py -M resnet50 -D Emotion6VA -B 32 -lr 2e-5
