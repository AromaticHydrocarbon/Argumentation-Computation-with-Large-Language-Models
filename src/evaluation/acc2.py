import re
import json
from collections import defaultdict
from tqdm import tqdm
import sys

original_stdout = sys.stdout


def extract_com(text):
    if not text.endswith('}'):
        return None,None
    # 正则表达式
    pattern = re.compile(r'"(IN|OUT|UNDEC)":\s*\[\s*((?:\[\s*(?:\d+(?:,\s*)?)*\s*](?:,\s*)?)*)\s*]')
    # 查找前三个匹配项
    matches = pattern.finditer(text)

    # 提取结果
    result = set()
    com = {}
    count = 0
    for match in matches:
        if count >= 3:
            break
        key = match.group(1)
        values = re.findall(r'\[\s*((?:\d+(?:,\s*)?)*)\s*]', match.group(2))
        try:
            inner_sets = [set(map(int, v.split(','))) if v else set() for v in values]
            outer_list = list(map(frozenset, inner_sets))
            if key in com:
                raise KeyError
                # result[key].extend(outer_list)
            else:
                com[key] = outer_list
        except ValueError:
            # 如果解析失败，返回 None
            print("ValueError")
            return set(), {}
        count += 1

    if "IN" not in com:
        return None, None
    if "UNDEC" not in com:
        return None, com['IN']
    com.setdefault('OUT', [])

    for i in range(len(com['IN'])):
        out_value = com['OUT'][i] if i < len(com['OUT']) else None
        undec_value = com['UNDEC'][i] if i < len(com['UNDEC']) else None
        result.add((com['IN'][i], out_value, undec_value))
    # print(result)
    return result, com['IN']


def extract_json(text):
    if not text.endswith('}'):
        return None
    IN_match = re.search(r'\"IN\": \[([0-9, ]+)\]', text)
    IN_list = []
    if IN_match:
        IN_parts = IN_match.group(1).split(',')
        IN_list = [int(num.strip()) for num in IN_parts if num.strip()]

    # 使用正则表达式提取 OUT 列表
    OUT_match = re.search(r'\"OUT\": \[([0-9, ]+)\]', text)
    OUT_list = []
    if OUT_match:
        OUT_parts = OUT_match.group(1).split(',')
        OUT_list = [int(num.strip()) for num in OUT_parts if num.strip()]

    # 使用正则表达式提取 UNDEC 列表
    UNDEC_match = re.search(r'\"UNDEC\": \[([0-9, ]+)\]', text)
    UNDEC_list = []
    if UNDEC_match:
        UNDEC_parts = UNDEC_match.group(1).split(',')
        UNDEC_list = [int(num.strip()) for num in UNDEC_parts if num.strip()]

    if (not IN_match and not OUT_match and not UNDEC_match):
        return None
    # 将列表转换为集合
    IN_set = set(IN_list)
    OUT_set = set(OUT_list)
    UNDEC_set = set(UNDEC_list)

    return {"IN": IN_set, "OUT": OUT_set, "UNDEC": UNDEC_set}

#
# if __name__ == '__main__':
#     with open('qwen.txt', 'w') as file1:
#         sys.stdout = file1
#         file_path = './generated_predictions.jsonl'
#         grd = {}
#         com = {}
#         grd['label'] = []
#         grd['predict'] = []
#         com['label'] = []
#         com['predict'] = []
#
#         # 先读取整个文件的内容到一个列表中
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#         i = 1
#         # 使用 tqdm 处理列表
#         for line in tqdm(lines, total=len(lines), desc="Processing lines"):
#             # 去除行末尾的换行符
#             line = line.strip()
#             if line:
#                 data = json.loads(line)
#                 re1 = ""
#                 re2 = ""
#                 print(i)
#                 if "solving the grounded extension of" in data['prompt']:
#                     grd['label'].append(extract_json(data['label']))
#                     grd['predict'].append(extract_json(data['predict']))
#                 elif "solving complete extensions of an abstract" in data['prompt']:
#                     # 处理数据
#                     com['label'].append(extract_com(data['label']))
#                     com['predict'].append(extract_com(data['predict']))
#                 i += 1
#
#         wrong = []
#         # 逐元素比较
#         grd_right = 0
#         grd_all = 0
#         j = 0
#         # print(len(grd['label']),len(grd['predict']))
#         for i in range(len(grd['label'])):
#             grd_all += 1
#             j += 1
#             if com['predict'][i] is not None and grd['label'][i]['IN'] == grd['predict'][i].get('IN', {}):
#                 grd_right += 1
#                 #
#                 # print(f"{j}: true")
#                 # print(grd['label'][i]['IN'])
#                 # print(grd['predict'][i].get('IN', {}))
#             else:
#                 print(f"{j}: false")
#                 print(grd['label'][i]['IN'])
#
#                 if grd['predict'][i] is not None:
#                     print(grd['predict'][i].get('IN', {}))
#                 else:
#                     print("None")
#
#         # print(wrong)
#
#         wrong = []
#         # 逐元素比较
#         right = 0
#         all = 0
#         for i in range(len(com['label'])):
#             all += 1
#             j += 1
#             if com['predict'][i] is not None and com['label'][i]['IN'] == com['predict'][i].get('IN', {}):
#                 right += 1
#                 #
#                 # print(f"{j}: true")
#                 # print(com['label'][i]['IN'])
#                 # print(com['predict'][i].get('IN', {}))
#             else:
#                 print(f"{j}: false")
#                 print(com['label'][i]['IN'])
#                 if com['predict'][i] is not None:
#                     print(com['predict'][i].get('IN', {}))
#                 else:
#                     print("None")
#                 wrong.append({"label": com['label'][i], "predict": com['predict'][i]})
#
#         # print(wrong)
#         print(f"grd_acc:{grd_right} / {grd_all}")
#         print(f"com_acc:{right} / {all}")
