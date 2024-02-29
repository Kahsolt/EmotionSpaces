@ECHO OFF

REM refine using EmoSet (the largest)
REM 88% / 73%
python train.py -D EmoSet -B 192 -E 20 -lr 2e-4

REM train other heads
REM 92% / 43%
python train.py -D Emotion6Dim7 -B 32 -E 100 -lr 2e-4 -L lightning_logs\version_2\checkpoints\epoch=13-step=10332.ckpt
python train.py -D Emotion6Dim6 -B 32 -E 4 -lr 0 0 2e-4 2e-4
python train.py -D Emotion6VA   -B 32 -E 4 -lr 0 0 2e-4 2e-4
python train.py -D TweeterI     -B 32 -E 4 -lr 0 0 2e-4 2e-4


REM TODO
REM refine using all together
REM python train.py -D EmoSet,Emotion6Dim7,Emotion6Dim6,Emotion6VA,TweeterI -B 192 -E 20 -lr 2e-4
