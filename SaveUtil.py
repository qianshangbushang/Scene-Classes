import numpy as np
import sys
import cv2
import os
import ParseUtil
import ImagePredeal as IP

def product_npz_data(image_path, label_path, label_json_path, save_path):
    if not os.path.exists(label_path):
        ParseUtil.parse_json_doc(label_json_path, label_path)
    image_result = []
    label_result = []
    f = open(label_path, "r+")
    lines = f.readlines()
    for line in lines[:10]:
        image_name = line.split(" ")[0]
        total_path = image_path + "/" + image_name
        print("deal with image :", total_path)
        image = cv2.imread(total_path)
        image = IP.resize(image)
        image_result.append(image)
        label_result.append(int(line.strip("\n").split(" ")[1]))

    image_result = np.asarray(image_result)
    label_result = np.asarray(label_result)
    np.savez(save_path, X=image_result, y=label_result)
    return image_result, label_result
    #print(image_result.shape)
    #print(image_result[0])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage：python product_npz_data.py image_base_path label_path")
        exit(-1)
    product_npz_data(sys.argv[1], sys.argv[2])