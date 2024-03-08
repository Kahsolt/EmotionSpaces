@ECHO OFF

SET MODEL=resnet50

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Plan A: refine on the largest dataset, then train the rest heads only

REM refine on EmoSet
python train.py -M %MODEL% -H mlp -D EmoSet -B 128 -E 20 -lr 2e-4

SET LOAD=lightning_logs\version_2\checkpoints\epoch=13-step=10332.ckpt

REM train other heads
python train.py -L %LOAD% -M %MODEL% -H mlp -D Emotion6Dim7 -B 32 -E 50 -lr 0 0 2e-4 2e-4
python train.py -L %LOAD% -M %MODEL% -H mlp -D Emotion6Dim6 -B 32 -E 50 -lr 0 0 2e-4 2e-4
python train.py -L %LOAD% -M %MODEL% -H mlp -D Emotion6VA   -B 32 -E 50 -lr 0 0 2e-4 2e-4
python train.py -L %LOAD% -M %MODEL% -H mlp -D TweeterI     -B 32 -E 50 -lr 0 0 2e-4 2e-4


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Plan B: refine on all datasets together

REM refine on all datasets
python train_all.py -M %MODEL% -H linear -B 32 -E 20 -lr 2e-4
