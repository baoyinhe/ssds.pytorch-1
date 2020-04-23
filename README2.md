# preparation in anaconda(python3.6)
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
# train
## prepare data
Remember to change the classes.
```
./prepare_data.py -s myVOC -d VOC2020 -p 1.0 -t 0.0 -c 1 
./prepare_data.py -s VOC2012 -d VOC2020 -p 0.02 -t 0.3
```
## choose a model
Choose a model and revise the configure file in ./experiments/cfgs/
Especially:

```
# NUM_CLASSES
Change NUM_CLASSES (also need to change classname in lib/dataset/voc.py) which corresponds to CLASSES in prepare_data.py

# RESUME_SCOPE
If use checkpoint and the NUM_CLASSES is different from the checkpoint model, change the RESUME_SCOPE:  from 'base,norm,extras,loc,conf' to 'base,norm,extras'.

# DATASET
Change the TRAIN_SETS and TEST_SETS according to your dataset's name.

# PHASE
Set the 'PHASE'='train'.
```
## start to train
```
python train.py --cfg=./experiments/cfgs/my_ssdlite_mobilenetv2_train_voc.yml
```
# test
Set the 'PHASE'='test'.
```
python test.py --cfg=./experiments/cfgs/my_ssdlite_mobilenetv2_train_voc.yml
```
