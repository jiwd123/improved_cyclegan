from PIL import Image
import os
import cv2
import numpy as np
import random

path = r'D:/github/cyclegan/output/A1/'
outpath = r'D:/github/cyclegan/output/C/'
orianpath = r'D:\github\cyclegan\datasets\blank2root\test\B/'
soilpath = r'D:/github/cyclegan/output/soil/'
x = os.listdir(path)
i=0
n=0
for i in x:
    print(x[n])
    name = x[n].strip('.png')
    img = cv2.imread(orianpath+name+'.jpg')
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.dilate(img, kernel)
    img1 = erosion-img
    img2 = cv2.bitwise_not(img1)
    img3=Image.fromarray(img2)
    img3 = img3.convert("RGBA")
    datas1 = img3.getdata()
    newData1 = []
    for item in datas1:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData1.append((255, 255, 255, 0))
        else:
            newData1.append((0, 0, 0, 50))
    img3.putdata(newData1)
    print('阴影完成'+x[n])
    to_image = Image.open(path+x[n])

    to_image = to_image.convert("RGBA")
    datas = to_image.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    to_image.putdata(newData)
    print('背景透明化完成'+x[n])
    to_image.paste(img3, (0,0), img3)
    print('叠加完成'+x[n])
    y = os.listdir(soilpath)
    filenames = "".join(random.sample(y, 1))
    print(filenames)
    soilimg = Image.open(soilpath+str(filenames))
    soilimg.paste(to_image, (0,0), to_image)

    soilimg.save(outpath+x[n],"PNG")
    print('土壤添加完成'+x[n])
    n=n+1