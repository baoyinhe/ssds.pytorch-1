## 1、测试精度低
测试时要把RESUME_CHECKPOINT改为自己训练好的模型，如：
RESUME_CHECKPOINT: './experiments/models/ssd_mobilenet_v2_voc/ssd_lite_mobilenet_v2_voc_epoch_290.pth'

另外，若之前使用预训练模型进行训练时没有加载'loc'和'conf'层，要在RESUME_SCOPE里添加上，因为测试的时候需要用完整的模型，如：
RESUME_SCOPE: 'base,norm,extras,loc,conf'
## 2、测试时出现‘KeyValueError'报错
测试前把./data/VOCdevkit/里的'annotations_cache'和'results'文件夹里的内容清空。