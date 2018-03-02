# -*- coding: utf-8 -*-
import cv2
import os
from PIL import Image
from get_va_labels import *
import scipy.misc as misc

data_path = '../AFEW-VA/data1.txt'


def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        center_x = x + width//2
        center_y = y + height//2
        result.append((center_x-150,center_y-150,center_x+150,center_y+150))
        # result.append((x,y,x+width,y+height))
    return result


def saveFaces(image_name):
    faces = detectFaces(image_name)
    print(len(faces))
    if faces:
        # 将人脸保存在save_dir目录下。
        # Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = image_name.split('/')[-1].split('.')[0]
        save_dir = save_dir.lstrip('F:\\')
        save_dir = "F:\\crop_data\\" + save_dir
        # os.mkdir(save_dir)
        file_name=save_dir.rstrip(save_dir.split('\\')[5])
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        count = 0
        for (x1,y1,x2,y2) in faces:
            print(x1,y1,x2,y2)
            image_crop_name = save_dir+".png"
            Image.open(image_name).crop((x1,y1,x2,y2)).resize((170,170),Image.ANTIALIAS).save(image_crop_name)
            count+=1
        if count > 1:
            f=open("F:\\multiImage.txt","w")
            f.write(image_name)
    else:
        f=open("F:\\noImage.txt","w")
        f.write(image_name)


# detect face and crop
path = 'E:/AFEW-VA\\01\\020'
with open(data_path, 'w') as file:
    findfiles(path, '.png', 1, file)

f = open(data_path)

for image_name in f.readlines():
    image_name = image_name.strip('\n').split(' ')[0]
    # saveFaces(image_name)
    save_dir = image_name.split('/')[-1].split('.')[0]
    save_dir = save_dir.lstrip('F:\\')
    save_dir = "F:\\crop_data\\" + save_dir
    # os.mkdir(save_dir)
    file_name = save_dir.rstrip(save_dir.split('\\')[5])
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    image_crop_name = save_dir + ".png"
    Image.open(image_name).crop((186 ,75 ,486, 375)).resize((170, 170), Image.ANTIALIAS).save(image_crop_name)




