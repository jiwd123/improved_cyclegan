import cv2
from PIL import Image
import os,shutil
img = cv2.imread(r"output/Improved_unet_seg.jpg")
target_size = 256
padding=(0, 0, 0)
max_y, max_x = img.shape[0], img.shape[1]
y0 = max_y
x0 = max_x
    # 若不能等分，则填充至等分
if max_x % target_size != 0:
    padding_x = target_size - (max_x % target_size)
    img = cv2.copyMakeBorder(img, 0, 0, 0, padding_x, cv2.BORDER_CONSTANT, value=padding)
    max_x = img.shape[1]
if max_y % target_size != 0:
    padding_y = target_size - (max_y % target_size)
    img = cv2.copyMakeBorder(img, 0, padding_y, 0, 0, cv2.BORDER_CONSTANT, value=padding)
    max_y = img.shape[0]
 
h_count = int(max_x / target_size)
v_count = int(max_y / target_size)
PATH = r'testdata/model7/'
IMAGES_PATH = PATH + '/A/' # 图片集地址
IMAGES_PATH2 = 'output\B/'
IMAGES_FORMAT = ['.png', '.png'] # 图片格式


IMAGE_SIZE = target_size            
IMAGE_ROW = h_count # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = v_count # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = PATH+'/B/jieguo.png' # 图片转换后的地址
IMAGE_SAVE_PATH2 = PATH +'/C/jieguo.jpg' #图片缓存地址
SAVE_PATH = PATH + '/jieguo.png'
to_image = Image.new('RGB', (IMAGE_ROW * IMAGE_SIZE, IMAGE_COLUMN * IMAGE_SIZE)) #创建一个新图
print(to_image.size)
print(max_x,max_y)
print(IMAGE_ROW,IMAGE_COLUMN)
indey = 0
x = os.listdir(IMAGES_PATH)
x.sort(key=lambda x:int(x[:-4]))
print(x[0])
print(x[2199])

for j in range(0, IMAGE_COLUMN):
    for i in range(0, IMAGE_ROW):
        from_image = Image.open(IMAGES_PATH + x[indey])
        to_image.paste(from_image, (i*IMAGE_SIZE, j*IMAGE_SIZE))
        indey = indey + 1
        
print(indey)
to_image = to_image.crop((0,0,x0,y0))
to_image.save(IMAGE_SAVE_PATH) # 保存新图
img= cv2.imread(IMAGE_SAVE_PATH)
cv2.imwrite(IMAGE_SAVE_PATH2,img)
to_image = Image.open(IMAGE_SAVE_PATH2)
to_image = to_image.convert("RGBA")
datas = to_image.getdata()
newData = []
for item in datas:
    if item[0] <= 91 and item[1] <= 81 and item[2] <= 72:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
to_image.putdata(newData)

to_image.save(SAVE_PATH,"PNG")
# shutil.rmtree(IMAGES_PATH)
# shutil.rmtree(IMAGES_PATH2)
# os.remove(IMAGE_SAVE_PATH)
# os.remove(IMAGE_SAVE_PATH2) #删除缓存