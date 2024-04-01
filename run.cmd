@ECHO OFF

SET MODEL=resnet50
SET DATASETS=EmoSet Emotion6Dim7 Emotion6Dim6 Emotion6VA TweeterI


:train
REM Step 1: refine a base ckpt with EmoSet (full weights)
python train.py -M %MODEL% -D EmoSet -lr 1e-4 --n_batch_train -1 --n_batch_valid -1

REM Step 2: train on mixed datasets (non-pretrained weights)
SET LOAD=lightning_logs\version_18\checkpoints\epoch=12-step=38376.ckpt
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 0 4e-5 1e-4

REM Step 3: refine on mixed datasets (full weights)
SET LOAD=lightning_logs\version_27\checkpoints\epoch=95-step=19200.ckpt
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 2e-6


:visualize
python vis.py -L "lightning_logs\version_27\checkpoints\epoch=95-step=19200.ckpt"
