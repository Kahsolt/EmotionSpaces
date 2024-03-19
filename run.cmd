@ECHO OFF

SET MODEL=resnet50

REM Step 1: refine a base ckpt with EmoSet (full weights)
python train.py -M %MODEL% -D EmoSet -lr 1e-4 --n_batch_train -1 --n_batch_valid -1


SET DATASETS=EmoSet Emotion6Dim7 Emotion6Dim6 Emotion6VA TweeterI

REM Step 2: train on mixed datasets (non-pretrained weights)
SET LOAD=lightning_logs\version_
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 0 1e-4 1e-4

REM Step 3: refine on mixed datasets (full weights)
SET LOAD=lightning_logs\version_
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 2e-6
