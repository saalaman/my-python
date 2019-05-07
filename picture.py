import numpy as np
from PIL import Image
import os
from skimage import io,transform,color
import pandas as pd
'''
第一步
该文件用来处理图像
对图像进行重命名、裁剪、缩小、通道转换等操作
'''
#测试集
images_dir = 'E:/Graduation-Project/Event&NoEvent/TEST_Normalized'
#训练集目录下有两个文件夹
train_path = 'E:/Graduation-Project/Event&NoEvent/Even_original'

#重命名
class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self,path):
        self.path = path#表示需要命名处理的文件夹
    def rename(self):
        filelist = os.listdir(self.path) #获取文件路径
        total_num = len(filelist) #获取文件长度（个数）
        #print(filelist,',',total_num)
        i = 1  #表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.png'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                # path_name.append(item.split(".")[0])
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), 'img'+str(i) + '.png')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>4s') + '.png')#这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d pngs' % (total_num, i))
#裁剪
def cut_image(path):
    count = os.listdir(path)
    print("count=", len(count))
    for item in count:
        src = os.path.join(os.path.abspath(path), item)
        im = Image.open(src)
        #去掉坐标轴的白边
        X1 = 80
        X2 = 575
        Y1 = 58
        Y2 = 426
        output_image = im.crop((X1,Y1,X2,Y2))
        output_image.save(src)
    print('finished')
#灰度化
def convert_to_gray(f, **args):
    image = io.imread(f)
    image = color.rgb2gray(image)
    return image
#尺寸归一化
def re_size(images_dir):
    count = os.listdir(images_dir)
    for item in count:
        src = os.path.join(os.path.abspath(images_dir), item)
        im = Image.open(src)
        im = im.resize((112,112))
        im.save(src)
        image = np.array(im)
        print(image.shape)
#通道归一化
def channels_normalization(images_dir):
    count = os.listdir(images_dir)
    for item in count:
        src = os.path.join(os.path.abspath(images_dir), item)
        im = Image.open(src)
        im = np.array(im)
        if len(im) == 2:
            print('1通道')
            c = []
            for i in range(3):
                c.append(im)
            im = np.asarray(c)
            im = im.transpose([1, 2, 0])
            im.save(src)
        elif im.shape[2] != 3:
            print('4通道')
            im = Image.open(src).convert("RGB")
            im.save(src)
        im = np.array(im)
        print(im.shape)
    print('finished channels_normalization ')

pd.set_option('display.width', 100)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

if __name__ == '__main__':
    #训练集处理
    list = os.listdir(train_path)
    for i in range(0, len(list)):
        path = os.path.join(train_path, list[i])
        demo = BatchRename(path)
        demo.rename()
        re_size(path)
        channels_normalization(path)
    #测试集处理
    demo = BatchRename(images_dir)
    demo.rename()
    cut_image(images_dir)
    re_size(images_dir)
    channels_normalization(images_dir)

