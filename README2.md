## preparation in anaconda(python3.6)
```
gpu:
conda install pytorch-gpu torchvision cudatoolkit=10.0
cpu:
conda install pytorch-cpu torchvision

conda install -c conda-forge tensorboardx 
conda install matplotlib opencv
conda install 'pillow<7.0.0'
conda install pyyaml
```
## train
Choose a model and revise the configure file in ./experiments/cfgs/

Set the 'PHASE'='train'.

Change classname in lib/dataset/voc.py
```
python train.py --cfg=./experiments/cfgs/ssd_lite_mobilenetv1_train_voc.yml
```
## test
Set the 'PHASE'='test'.
```
python test.py --cfg=./experiments/cfgs/ssd_lite_mobilenetv1_train_voc.yml
```
