#!/usr/bin/python3
import os
import shutil
import random
import argparse
from xml.etree import ElementTree

# class to keep
CLASSES = ('background',
           'bus', 'car',
           'bicycle', 'motorbike', 'person',
           'front_wheel', 'back_wheel', 'door')

# remove objects which are not in CLASSES
def handle(imgID,src,dest):
    global basedir
    tree = ElementTree.parse(basedir + src + "/Annotations/" + imgID + ".xml")
    root = tree.getroot()

    haveobj = False
    for obj in root.findall('object'):
        obj_cls = obj.find('name').text
        if(obj_cls not in CLASSES):
            # print("remove "+ obj_cls)
            root.remove(obj)
        else:
            haveobj = True
    if haveobj:
        print("copy "+imgID)
        tree.write(basedir + dest + "/Annotations/" + imgID + ".xml")
        try:
            shutil.copyfile(basedir + src +"/JPEGImages/"+imgID+".jpg",basedir + dest + "/JPEGImages/"+imgID+".jpg")
        except FileNotFoundError:
            shutil.copyfile(basedir + src +"/JPEGImages/"+imgID+".JPG",basedir + dest + "/JPEGImages/"+imgID+".jpg")
        
        r = random.uniform(0,1)
        if r <= args.test:
            test.write(imgID +'\n')
            print("=== Add %s to test" %(imgID))
        else:
            trainval.write(imgID +'\n')
            print("--- Add %s to trainval" %(imgID))

# parse args
parser = argparse.ArgumentParser(description="""\
    Remove unkown labels from annotation, then copy xml and jpg to destation.
    Example:
    ./prepare_data.py -s myVOC20200114 -d VOC2020 -p 1.0
    """)
parser.add_argument("-s","--src",default="VOC2007",help="source subdir in ~/data/VOCdevkit/")
parser.add_argument("-d","--dest",default="VOC2020",help="destination subdir in ~/data/VOCdevkit/")
parser.add_argument("-p","--prop",type=float,default=1.0,help="probility to copy")
parser.add_argument("-t","--test",type=float,default=0.3,help="percent for test")
parser.add_argument("-c","--clear",type=bool,default=0,help="clear the file")
args = parser.parse_args()
# print(args.src,args.dest,args.prop)

basedir = os.path.expanduser('~')+"/data/VOCdevkit/"
srcdir = basedir + args.src + "/Annotations/"
destdir = basedir + args.dest + "/Annotations/"

trainval = open(basedir + args.dest + "/ImageSets/Main/trainval.txt",'a+')
test = open(basedir + args.dest + "/ImageSets/Main/test.txt",'a+')


print("1. Clear the trainval.txt, test.txt and 'VOC'file.")
if args.clear:
    trainval.seek(0)
    trainval.truncate()
    test.seek(0)   
    test.truncate()
    i, j = 0, 0
    for file in os.listdir(basedir + args.dest + "/JPEGImages/"):
        path_file = os.path.join(basedir + args.dest + "/JPEGImages/", file)
        os.remove(path_file)
        i += 1
    print('Already remove %d images.' %i)
    for file in os.listdir(destdir):
        path_file = os.path.join(destdir, file)
        os.remove(path_file)
        j += 1
    print('Already remove %d xml.' %j)
else:
    print('No need to clear.')


# handle xml and copy
print("2. Walk src annotation directory " + srcdir)
for file in os.listdir(srcdir):
    filename = file.split(".")[0]
    r = random.uniform(0,1)
    if r <= args.prop:
        handle(filename, args.src, args.dest)
 
trainval.close()
test.close()
print("Update destination trainval.txt and test.txt")


