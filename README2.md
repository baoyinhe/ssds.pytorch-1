## preparation in anaconda
```
conda install pytorch-gpu torchvision cudatoolkit=10.0
conda install -c conda-forge tensorboardx 
conda install matplotlib opencv
conda install 'pillow<7.0.0'
conda install pyyaml
```
## train
Choose a model and revise the configure file in ./experiments/cfgs/

Set the 'PHASE'='train'.
```
python train.py --cfg=./experiments/cfgs/ssd_lite_mobilenetv1_train_voc.yml
```
## test
Set the 'PHASE'='test'.
```
python test.py --cfg=./experiments/cfgs/ssd_lite_mobilenetv1_train_voc.yml
```
