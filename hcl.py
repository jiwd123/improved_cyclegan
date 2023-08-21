import cv2
import os
from PIL import Image
#后处理
xa = os.listdir('testdata\model1\A/')
ia=0
na=0
for ia in xa:
    name=xa[na].strip('.png')
    img = Image.open('testdata\model1\A/'+xa[na])
    img1 = Image.open(r'datasets\blank2shuailao\test\B/'+name+'.jpg')
    
    for x in range(img.width):
        for y in range(img.height):
            RGB=img1.getpixel((x,y))
            #print(RGB)
            if RGB<=(127,127,127):
                img.putpixel((x,y),(255,255,255))
    img.save('testdata\model1\E/'+xa[na])
    print('处理完成'+'   ' + 'A/'+xa[na])
    na=na+1
    
# xb = os.listdir('output/B/')
# ib=0
# nb=0
# for ib in xb:
#     name = xb[nb].strip('.png')
#     img = Image.open('output/B/'+xb[nb])
#     img1 = Image.open(r'datasets\blank2root\test\B/'+name+'.jpg')
#     newData1 = []
#     datas1 = img1.getdata()
#     for item in datas1:
#         if item[0] == 0 and item[1] == 0 and item[2] == 0:
#             newData1.append((0, 0, 0))
#         else:
#             newData1.append((255, 255, 255))
#     img.putdata(newData1)
#     img.save('output/B1/'+xb[nb])
#     print('处理完成'+'   ' + 'B/'+xb[nb])
#     nb=nb+1