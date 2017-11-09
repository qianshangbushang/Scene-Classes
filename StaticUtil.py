import json


def find_class_count(json_path = "scene_train_annotations_20170904.json"):
    dict = {}
    f = open(json_path, encoding='utf-8')
    docs = json.load(f)
    for doc in docs:
        if not (doc['label_id']) in dict:
            dict[doc['label_id']] = 1
        else:
            dict[doc['label_id']] += 1
    for (d, k) in dict.items():
        print(d, k)
    print(len(dict))
    return

if __name__ == '__main__':
    find_class_count()