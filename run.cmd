@ECHO OFF

SET MODEL=resnet50

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Plan A: refine on the largest dataset, then train the rest heads only

REM train on EmoSet (non-pretrained weights)
python train.py -M %MODEL% -D EmoSet -B 32 -E 100 -lr 0 1e-3 1e-3
REM refine on EmoSet (full weights)
SET LOAD=lightning_logs\version_0\checkpoints\
python train.py -L %LOAD% -M %MODEL% -D EmoSet -B 32 -E 100 -lr 2e-4

REM train other heads
SET LOAD=lightning_logs\version_1\checkpoints\
python train.py -L %LOAD% -M %MODEL% -D Emotion6Dim7 -B 32 -E 50 -lr 0 0 1e-3
python train.py -L %LOAD% -M %MODEL% -D Emotion6Dim6 -B 32 -E 50 -lr 0 0 1e-3
python train.py -L %LOAD% -M %MODEL% -D Emotion6VA   -B 32 -E 50 -lr 0 0 1e-3
python train.py -L %LOAD% -M %MODEL% -D TweeterI     -B 32 -E 50 -lr 0 0 1e-3


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Plan B: refine on all datasets together

SET DATASETS=EmoSet Emotion6Dim7 Emotion6Dim6 Emotion6VA TweeterI

REM train on all datasets (non-pretrained weights)
python train_all.py -M %MODEL% -D %DATASETS% -B 32 -E 100 -lr 0 1e-3 1e-3
REM refine on all datasets (full weights)
SET LOAD=lightning_logs\version_2\checkpoints\
python train_all.py -L %LOAD% -M %MODEL% -D %DATASETS% -B 32 -E 100 -lr 2e-4
