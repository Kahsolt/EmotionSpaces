@ECHO OFF

REM refine using EmoSet (the largest)
python train.py -D EmoSet -B 192 -E 50 -lr 2e-4 --load lightning_logs\version_0\checkpoints\epoch=9-step=4930.ckpt
