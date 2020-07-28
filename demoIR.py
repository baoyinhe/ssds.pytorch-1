from __future__ import print_function
import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2

from lib.layers.functions.detection import Detect
from lib.ssds_vino import ObjectDetector
from lib.utils.config_parse import cfg_from_file

# VOC_CLASSES = ( 'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = ('bus', 'car',
    'bicycle', 'motorbike', 'person',
    'front_wheel', 'back_wheel', 'door')

# # 类别标签变量.
# classNames = { 0: 'background',
#     1: 'bus', 2: 'car', 3: 'bicycle', 4: 'motorbike',
#     5: 'person', 6: 'front_wheel', 7: 'back_wheel', 8: 'door'}

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('--cfg', dest='confg_file',
            help='the address of optional config file', default=None, type=str, required=True)
    parser.add_argument('--demo', dest='demo_file',
            help='the address of the demo file', default=None, type=str, required=True)
    parser.add_argument('-t', '--type', dest='type',
            help='the type of the demo file, could be "image", "video", "camera" or "time", default is "image"', default='image', type=str)
    # parser.add_argument('-d', '--display', dest='display',
    #         help='whether display the detection result, default is True', default=True, type=bool)
    # parser.add_argument('-s', '--save', dest='save',
    #         help='whether write the detection result, default is False', default=False, type=bool)  
    parser.add_argument("--model", dest='model', type=str, default="MobileNetSSD_deploy",
                               help="path to trained model")
    parser.add_argument("--thr", default=0.8, type=float, help="confidence threshold to filter out weak detections")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


# COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# FONT = cv2.FONT_HERSHEY_SIMPLEX

def demo(args, image_path):

    input_size = (300, 300)

    # 加载模型 
    net = cv2.dnn.Net_readFromModelOptimizer(args.model+".xml", args.model+".bin")
    # 设置推理引擎后端
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # 设置运算设备
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    # 打开视频文件或摄像头 
    if args.demo_file:
        cap = cv2.VideoCapture(args.demo_file)
    else:
        cap = cv2.VideoCapture(0)

    # while True:

    # 读取一帧图像
    ret, frame = cap.read()
    # 将图片转换成模型输入
    blob = cv2.dnn.blobFromImage(frame, 0.007843, input_size, (127.5, 127.5, 127.5), False)

    # 转换后的待输入对象blob设置为网络输入
    net.setInput(blob)

    # # 开始进行网络推理运算
    # out = net.forward()

    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector(net)

    # # 3. load image
    # image = cv2.imread(image_path)

    # 4. detect
    _labels, _scores, _coords = object_detector.predict(frame)

    # 5. draw bounding box on the image
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
        cv2.putText(frame, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
    # 6. visualize result
    if args.display is True:
        cv2.imshow('result', frame)
        cv2.waitKey(0)

    # 7. write result
    if args.save is True:
        path, _ = os.path.splitext(image_path)
        cv2.imwrite(path + '_result.jpg', frame)

        # # 获取输入图像尺寸(300x300)
        # cols = input_size[1]
        # rows = input_size[0]
        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2] # 目标对象置信度 
        #     if confidence > args.thr: # Filter prediction
        #         class_id = int(detections[0, 0, i, 1]) # 目标对象类别标签

        # # 目标位置
        # xLeftBottom = int(detections[0, 0, i, 3] * cols)
        # yLeftBottom = int(detections[0, 0, i, 4] * rows)
        # xRightTop   = int(detections[0, 0, i, 5] * cols)
        # yRightTop   = int(detections[0, 0, i, 6] * rows)
        # # 变换尺度
        # heightFactor = frame.shape[0]/300.0  
        # widthFactor = frame.shape[1]/300.0
        # # 获取目标实际坐标
        # xLeftBottom = int(widthFactor * xLeftBottom)
        # yLeftBottom = int(heightFactor * yLeftBottom)
        # xRightTop   = int(widthFactor * xRightTop)
        # yRightTop   = int(heightFactor * yRightTop)

        # # 框出目标对象
        # cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
        #                         (0, 255, 0))
        # # 标记标签和置信度
        # if class_id in classNames:
        #     label = classNames[class_id] + ": " + str(confidence)
        #     labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        
        #     yLeftBottom = max(yLeftBottom, labelSize[1])
        #     cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
        #                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
        #                         (255, 255, 255), cv2.FILLED)
        #     cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        #     print(label) # 输出类别和置信度

        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"): # 按下q键退出程序
        #     break
        # if key == ord("s"): # 按下s键保存检测图像
        #     cv2.imwrite('detection.jpg', frame)

    
    

def demo_live(args, video_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load video
    video = cv2.VideoCapture(video_path)

    index = -1
    while(video.isOpened()):
        index = index + 1
        sys.stdout.write('Process image: {} \r'.format(index))
        sys.stdout.flush()

        # 4. read image
        flag, image = video.read()
        if flag == False:
            print("Can not read image in Frame : {}".format(index))
            break

        # 5. detect
        _labels, _scores, _coords = object_detector.predict(image)

        # 6. draw bounding box on the image
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
        # 7. visualize result
        if args.display is True:
            cv2.imshow('result', image)
            cv2.waitKey(33)

        # 8. write result
        if args.save is True:
            path, _ = os.path.splitext(video_path)
            path = path + '_result'
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(path + '/{}.jpg'.format(index), image)        


def time_benchmark(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load image
    image = cv2.imread(image_path)

    # 4. time test
    warmup = 20
    time_iter = 100
    print('Warmup the detector...')
    _t = list()
    for i in range(warmup+time_iter):
        _, _, _, (total_time, preprocess_time, net_forward_time, detect_time, output_time) \
            = object_detector.predict(image, check_time=True)
        if i > warmup:
            _t.append([total_time, preprocess_time, net_forward_time, detect_time, output_time])
            if i % 20 == 0: 
                print('In {}\{}, total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
                    i-warmup, time_iter, total_time, preprocess_time, net_forward_time, detect_time, output_time
                ))
    total_time, preprocess_time, net_forward_time, detect_time, output_time = np.sum(_t, axis=0)/time_iter * 1000 # 1000ms to 1s
    print('In average, total time: {}ms \n preprocess: {}ms \n net_forward: {}ms \n detect: {}ms \n output: {}ms'.format(
        total_time, preprocess_time, net_forward_time, detect_time, output_time
    ))
    with open('./time_benchmark.csv', 'a') as f:
        f.write("{:s},{:.2f}ms,{:.2f}ms,{:.2f}ms,{:.2f}ms,{:.2f}ms\n".format(args.confg_file, total_time, preprocess_time, net_forward_time, detect_time, output_time))


    
if __name__ == '__main__':
    args = parse_args()
    if args.type == 'image':
        demo(args, args.demo_file)
    elif args.type == 'video':
        demo_live(args, args.demo_file)
    elif args.type == 'camera':
        demo_live(args, int(args.demo_file))
    elif args.type == 'time':
        time_benchmark(args, args.demo_file)
    else:
        AssertionError('type is not correct')
