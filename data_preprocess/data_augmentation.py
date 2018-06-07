import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter
import numpy as np
import random


def Data_augmentation(src_path,dest_path):
    new_context=[]
    file=src_path+'test.txt'
    with open(file,'r') as f:
        context=f.readlines()
        new_context=context
        for i in context:
            image=i.strip().split(' ')[0].replace('./','')
            image_path=src_path+image
            label=i.strip().split(' ')[1]
            new_image=generate_new_image(image_path)
            new_path=dest_path+image.split('.')[0]+'.png'
            new_image.save(new_path)
            # string='./'+image.split('.')[0]+"_new"+'.png'+' '+label
            # new_context.append(string)
    # with open ('/home/dilligencer/code/比赛/ocr---AI/competition_final/train_new.txt','w') as f:
    #     f.writelines(new_context)

def generate_new_image(image_path):
    image=Image.open(image_path)
    # random_angle=np.random.randint(-3,3)
    # image=image.rotate(random_angle,Image.BICUBIC) # 旋转
    # image=image.filter(ImageFilter.BLUR)  # 加模糊
    # random_factor = np.random.randint(0, 11) / 10  # 随机因子
    # color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(7, 9) / 10 # 亮度增强的倍数
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10,12) / 10.  # 随机因1子
    image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    # random_factor = np.random.randint(0, 11) / 10.  # 随机因子
    # image=ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    def gaussianNoisy(im, mean=0, sigma=0.3):
        """
        :param im:单通道图像
        :param mean:
        :param sigma:
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten())
    img_g = gaussianNoisy(img[:, :, 1].flatten())
    img_b = gaussianNoisy(img[:, :, 2].flatten())
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))



def make_new_train_txt(src_file,des_file):
    with open(src_file,'r') as f:
        context = f.readlines()
        new_context = []
        for i in context:
            image= i.strip().split(' ')[0]
            content=i.strip().split(' ')[1]
            new_image=image.split('.png')[0]+'_new_1'+'.png'
            line=new_image+' '+content+'\n'
            new_context.append(line)
    with open(des_file,'w') as des_file:
        des_file.writelines(new_context)



def main():
    Data_augmentation('/home/dilligencer/code/比赛/ocr---AI/competition_final/test/',
                      '/home/dilligencer/code/比赛/ocr---AI/competition_final/test_new/')
    # make_new_train_txt('/home/dilligencer/code/比赛/ocr---AI/train/train.txt',
    #                    '/home/dilligencer/code/比赛/ocr---AI/train_new/train_new.txt')



if __name__ == '__main__':
    main()