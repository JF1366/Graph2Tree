# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import os

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def write_json(path,file):
    with open(path,'w') as f:
        json.dump(file,f)
        
def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    # 初始化一个空字符串 js，用于累积读取的行
    js = ""
    # 一个data列表用于存储解析后的JSON对象
    data = []
    for i, s in enumerate(f):
        js += s # 将当前行 s 添加到 js 字符串的末尾。
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            # 使用 json.loads 函数将累积的字符串转换为一个Python字典 data_d
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                # 将这个子字符串从 equation 字符串中移除
                data_d["equation"] = data_d["equation"][:-5]
            # 将解析后的字典 data_d 添加到列表 data 中
            data.append(data_d)
            # 并重置 js 字符串为空，以便开始解析下一个JSON对象。
            js = ""
    # 函数返回包含所有解析后JSON对象的列表 data
    return data

def combine_json(graph_dict, ori_json):
    new_data = []
    for i in tqdm(range(len(ori_json))):
        item = ori_json[i]
        item['group_num'] = graph_dict[item['id']]['group_num']
        new_data.append(item)        
    print('Graph has been inserted into ori_json!')
    return new_data

def main(whole_path, save_path, ori_path):
    whole = read_json(whole_path)
    whole_dict = dict([(item['id'],item) for item in whole])
    ori = load_raw_data(ori_path)
    new_data = combine_json(whole_dict, ori)
    write_json(save_path, new_data)
    print('Finished')
    
whole_path = 'E:\MySelf_WorkCodes\PyCharm\Research_direction_code\Graph2Tree-master\math23k\data\whole_processed.json'
ori_path = 'E:\MySelf_WorkCodes\PyCharm\Research_direction_code\Graph2Tree-master\math23k\data\Math_23K.json'
save_path = 'E:\MySelf_WorkCodes\PyCharm\Research_direction_code\Graph2Tree-master\math23k\data\Math_23K_processed.json'
main(whole_path, save_path, ori_path)


