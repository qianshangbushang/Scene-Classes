# coding=utf-8
import cv2
import numpy as np
import os
import glob
from PIL import Image

image_root = "F:/PyCharm Code/challengerAi/data"


def getImage():
    image_data = []
    image_num = 7
    for i in range(1, image_num):
        img = Image.open(image_root + '/' + str(i) + '.jpg')
        img.show()


def resize(image, inter=cv2.INTER_AREA):  # inter 为插值的方式
    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[:2]
    # 找到最长边，根据最长边进行缩放
    if h > w:
        # 根据高度计算缩放比例
        r = 224 / float(h)
        dim = (int(w * r), 224)
    else:
        # 根据宽度计算缩放比例
        r = 224 / float(w)
        dim = (224, int(h * r))

    # 缩放图像
    img_resize = cv2.resize(image, dim, interpolation=inter)
    img_c = np.zeros((224, 224, 3), np.uint8)
    (h_c, w_c) = img_resize.shape[:2]

    if h_c < w_c:
        l1 = int(112 - h_c / 2)
        l2 = int(112 + h_c / 2)
        print(l1, l2)
        img_c[l1:l2, 0:224] = img_resize
    else:
        l1 = int(112 - w_c / 2)
        l2 = int(112 + w_c / 2)
        print(l1, l2)
        img_c[0:224, l1:l2] = img_resize
    return img_c


def run(input_path, output_path):

    print(input_path+"\\*.jpg")
    files = glob.glob(input_path + "\\*.*")
    print(files)
    for i in files:
        print("dealing with image: " + (i))
        img_src = cv2.imread(i)
        #cv2.imshow("src", img_src)
        img_resize = resize(img_src)
        #cv2.imshow("R", img_resize)
        img_c = np.zeros((224, 224, 3), np.uint8)
        (h_c, w_c) = img_resize.shape[:2]

        if h_c < w_c:
            l1 = int(112 - h_c / 2)
            l2 = int(112 + h_c / 2)
            print(l1, l2)
            img_c[l1:l2, 0:224] = img_resize
        else:
            l1 = int(112 - w_c / 2)
            l2 = int(112 + w_c / 2)
            print(l1, l2)
            img_c[0:224, l1:l2] = img_resize
        cv2.imshow("rc", img_c)
        cv2.waitKey(20)
        name = (i.split("\\")[-1]).split(".")[0]
        cv2.imwrite(output_path + '/' + name + '_r.jpg', img_c)


#################################
# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) != 3:
#         print("usage: python image_data_process.py input_image_dir, output_image_dir")
#         print(sys.argv)
#         #exit(-1)
#     print(sys.argv)
#     #run(r"D:\data_img\src_img", r"D:\data_img\processed")
#     run(sys.argv[1], sys.argv[2])
#     exit(0)
if __name__ == '__main__':
    image = cv2.imread("1.jpg")
    cv2.imshow("1",image)
    image = resize(image)
    cv2.imshow("2",image)
    cv2.waitKey()
    print(image.shape)