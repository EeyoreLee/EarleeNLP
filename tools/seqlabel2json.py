# -*- encoding: utf-8 -*-
'''
@create_time: 2022/08/26 10:20:10
@author: lichunyu
'''
import os
import json
import argparse

def main(path):
    with open(path, "r") as f:
        data = f.read().splitlines()
    data = [_.split("\t") for _ in data]
    if len(data[0][0]) > 1:
        data = [[_[0][0], _[1]]  if len(_)>1 else _ for _ in data]
    data4j = []
    text = []
    seq_label = []
    for i in data:
        if i[0] == "":
            data4j.append({"sentence": text, "ner": label2type(seq_label)})
            text = []
            seq_label = []
            continue
        text.append(i[0])
        seq_label.append(i[1])
    json_filename = path + ".json"
    with open(json_filename, "w") as f:
        f.write(json.dumps(data4j, ensure_ascii=False, indent=4))
    return json_filename

def label2type(label:list):
    result=[]
    ids = []
    for idx, i in enumerate(label):
        if i == "O":
            if ids:
                result.append({"index": ids, "type": tag})
                ids = []
            continue
        if i.startswith("B-"):
            if ids:
                result.append({"index": ids, "type": tag})
                ids = []
        ids.append(idx)
        tag = i[2:]
        if idx == len(label)-1:
            if ids:
                result.append({"index": ids, "type": tag})
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    path = args.file
    if path is None:
        print("processing failed...")
        print("please setup the arg called --file")
        exit(1)
    processed_filename = main(path)
    print(f"content is processed. the processed filename is \n {processed_filename}")