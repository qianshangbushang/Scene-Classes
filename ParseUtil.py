import os
import json

def parse_config():
    if not os.path.exists("./configure"):
        print("Failed to find the configure document!")
        exit(-1)
    dict = {}
    for line in open("./configure"):
        arr = line.strip("\n").split("=", 2)
        dict[arr[0]] = arr[1]
    return dict


def parse_json_doc(json_path, output_file):
    f = open(json_path, encoding='utf-8')
    f_out = open(output_file, "w+")
    docs = json.load(f)
    for doc in docs:
        image_id = doc['image_id']
        label_id = doc['label_id']
        print(image_id, label_id)
        f_out.write(image_id + " " + label_id + "\n")



if __name__ == '__main__':
    dict = parse_config()
    print(dict)