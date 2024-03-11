@ECHO OFF

SET MODEL=resnet50

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Plan A: refine on the largest dataset, then train the rest heads only

REM train on EmoSet (non-pretrained weights)
python train.py -M %MODEL% -D EmoSet -lr 0 1e-5 1e-5
REM refine on EmoSet (full weights)
SET LOAD=lightning_logs\version_0\checkpoints\
python train.py -L %LOAD% -M %MODEL% -D EmoSet -lr 1e-5

REM train other heads
SET LOAD=lightning_logs\version_1\checkpoints\
python train.py -L %LOAD% -M %MODEL% -D Emotion6Dim7 -lr 0 0 1e-5
python train.py -L %LOAD% -M %MODEL% -D Emotion6Dim6 -lr 0 0 1e-5
python train.py -L %LOAD% -M %MODEL% -D Emotion6VA   -lr 0 0 1e-5
python train.py -L %LOAD% -M %MODEL% -D TweeterI     -lr 0 0 1e-5


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Plan B: refine on mixed datasets together

SET DATASETS=EmoSet Emotion6Dim7 Emotion6Dim6 Emotion6VA TweeterI

REM train on mixed datasets (non-pretrained weights)
python train.py -M %MODEL% -D %DATASETS% -lr 0 1e-5 1e-5
REM refine on mixed datasets (full weights)
SET LOAD=lightning_logs\version_0\checkpoints\epoch=30-step=6200.ckpt
python train.py -L %LOAD% -M %MODEL% -D %DATASETS% -lr 1e-5
