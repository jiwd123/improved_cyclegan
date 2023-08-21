import cv2
import os
from PIL import Image
#后处理
xb= os.listdir('testdata\model1\A/')
ib=0
nb=0
for ib in xb:
    name = xb[nb].strip('.png')
    img = Image.open('testdata\model1\A/'+xb[nb])
    for x in range(img.width):
        for y in range(img.height):
            RGB=img.getpixel((x,y))
            #print(RGB)
            if RGB<=(127,127,127):
                img.putpixel((x,y),(255,255,255))
    img.save('testdata\model1\D/'+xb[nb])
    print('处理完成'+'   ' + 'A2/'+xb[nb])
    nb=nb+1