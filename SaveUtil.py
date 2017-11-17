import numpy as np
import sys
import cv2
import os
import ParseUtil
import ImagePredeal as IP
from keras.preprocessing.image import *
import glob


def generate_data():
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=10
    )

def test():
    image = cv2.imread("3.jpg")
    for i in range(5):
        point_arr = np.random.randint(low=(0.0 * (image.shape[0] + image.shape[1])),
                                      high=int(0.2 * (image.shape[0] + image.shape[1])),
                                      size=2)
        shape_arr = np.random.randint(low=int(0.25 * (image.shape[0] + image.shape[1])),
                                      high=int(0.4 * (image.shape[0] + image.shape[1])),
                                      size=2)
        print("image_size; ", image.shape)
        # w, h = int(0.9*image.shape[1]), int(0.9*image.shape[0])
        #w, h = shape_arr[0], shape_arr[1]
        w = shape_arr[0] if shape_arr[0] < image.shape[1] else int(0.5 * image.shape[1])
        h = shape_arr[1] if shape_arr[1] < image.shape[0] else int(0.5 * image.shape[0])
        x = point_arr[0] if point_arr[0] < image.shape[1] else int(0.1 * image.shape[1])
        y = point_arr[1] if point_arr[1] < image.shape[0] else int(0.1 * image.shape[0])
        #x, y = point_arr[0], point_arr[1]
        end_x = image.shape[1] if x + w > image.shape[1] else x + w
        end_y = image.shape[0] if y + h > image.shape[0] else y + h
        if y > end_y:
            end_y, y = y, end_y
        if x > end_x:
            end_x, x = x, end_x
        print("size: ", end_y -y, end_x - x)
        print(end_y, y)
        temp_image = IP.resize(image[y:end_y, x:end_x], 112)
    return temp_image
        #cv2.imwrite("../output_file/" + label_name + "_" + image_name.strip(".jpg") + "_" + str(i) + ".jpg", temp_image)
        #image_result.append(img_to_array(temp_image))
        #label_result.append(int(label_name))

def product_npz_data2(image_path, label_path, label_json_path, save_path):
    if not os.path.exists(label_path):
        ParseUtil.parse_json_doc(label_json_path, label_path)
    image_result = []
    label_result = []
    f = open(label_path, "r+")
    lines = f.readlines()
    for line in lines[:10]:
        image_name = line.split(" ")[0]
        label_name = line.strip("\n").split(" ")[1]
        total_path = image_path + "/" + image_name
        print("deal with image :", total_path)
        image = cv2.imread(total_path)
        for i in range(5):
            point_arr = np.random.randint(low=(0.0 * (image.shape[0]+image.shape[1])),
                                          high=int(0.2 * (image.shape[0]+image.shape[1])),
                                          size=2)
            shape_arr = np.random.randint(low=int(0.25 * (image.shape[0] + image.shape[1])),
                                          high=int(0.4 * (image.shape[0] + image.shape[1])),
                                          size=2)
            print("image_size; ", image.shape)
            #w, h = int(0.9*image.shape[1]), int(0.9*image.shape[0])
            w = shape_arr[0] if shape_arr[0] < image.shape[1] else int(0.5 * image.shape[1])
            h = shape_arr[1] if shape_arr[1] < image.shape[0] else int(0.5 * image.shape[0])
            x = point_arr[0] if point_arr[0] < image.shape[1] else int(0.1 * image.shape[1])
            y = point_arr[1] if point_arr[1] < image.shape[0] else int(0.1 * image.shape[0])
            end_x = image.shape[1] if x + w > image.shape[1] else x + w
            end_y = image.shape[0] if y + h > image.shape[0] else y + h
            if y > end_y :
                end_y, y = y, end_y
            if x > end_x:
                end_x, x = x, end_x
            print("size: ", end_y-y, end_x-x)
            temp_image = IP.resize(image[y:end_y, x:end_x], 224)
            if not os.path.exists("../output_file/" + label_name):
                os.mkdir("../output_file/" + label_name)
            cv2.imwrite("../output_file/"+label_name+"_"+image_name.strip(".jpg") + "_" + str(i) + ".jpg", temp_image)
            image_result.append(img_to_array(temp_image))
            label_result.append(int(label_name))
    image_result = np.asarray(image_result)
    label_result = np.asarray(label_result)
    np.savez(save_path, X=image_result, y=label_result)
    return image_result, label_result


def product_npz_data(image_path, label_path, label_json_path, save_path):
    if not os.path.exists(label_path):
        ParseUtil.parse_json_doc(label_json_path, label_path)
    image_result = []
    label_result = []
    f = open(label_path, "r+")
    lines = f.readlines()
    for line in lines:
        image_name = line.split(" ")[0]
        label_name = line.split(" ")[1]
        total_path = image_path + "/" + image_name
        print("deal with image :", total_path)
        image = cv2.imread(total_path)
        image = IP.resize(image, 112)
        if not os.path.exists("../output_file/" + label_name):
            os.mkdir("../output_file/" + label_name)
        cv2.imwrite("../output_file/"+label_name+"/"+image_name, image)
        image_result.append(img_to_array(image))
        label_result.append(int(line.strip("\n").split(" ")[1]))

    image_result = np.asarray(image_result)
    label_result = np.asarray(label_result)
    np.savez(save_path, X=image_result, y=label_result)
    return image_result, label_result
    #print(image_result.shape)
    #print(image_result[0])


if __name__ == '__main__':
    dict = ParseUtil.parse_config()
    print(dict)
    image_base_path = dict['train_image_path']
    label_path = dict['train_label_path']
    label_json_path = dict['train_json_path']
    save_path = "./train.npz"
    product_npz_data2(image_base_path, label_path, label_json_path, save_path)
    #product_npz_data2("D://")
    #test()
