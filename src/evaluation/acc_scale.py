import json
from tqdm import tqdm
from pathlib import Path
from acc2 import extract_json, extract_com

root_dir = Path(__file__).parent
# 假设 lines 是已经加载的数据行列表
# 初始化数据结构
qwen2_exp = ["13474"]
qwen2_no_exp = ["15428"]
dirs = ["scale"]
llama3_exp_dirs = ["1684", "3368", "5052", "6737", "8421", "10105", "11789", "13472"]

llama3_no_exp_dirs = ["13496", "11812", "10125", "8437", "6750", "5062", "3375", "1687"]
for dir in dirs:
    # file_path = root_dir / "llama3" / "no_exp" / dir / 'generated_predictions.jsonl'
    file_path = root_dir / "qwen2" / "exp" / dir / 'generated_predictions.jsonl'
    grd = {'label': [], 'predict': []}
    com = {'label': [], 'predict': []}
    stb = {'label': [], 'predict': []}
    prf = {'label': [], 'predict': []}
    # 先读取整个文件的内容到一个列表中
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 解析数据
    for idx, line in enumerate(tqdm(lines, total=len(lines), desc="Processing lines")):
        line = line.strip()
        if line:
            data = json.loads(line)
            # print(idx)
            group = idx // 100
            if group % 2 == 0:
                label = extract_json(data['label'])
                predcit = extract_json(data['predict'])
                if label is None:
                    grd['label'].append(None)
                    grd['predict'].append(None)
                else:
                    grd['label'].append(label)
                    grd['predict'].append(predcit)
            # 第2101至4200条是 com 类型
            else:
                # com extension
                result_l, com_l = extract_com(data['label'])
                result_p, com_p = extract_com(data['predict'])
                com['label'].append(com_l)
                com['predict'].append(com_p)
                if result_l is None:
                    prf['label'].append(None)
                    prf['predict'].append(None)
                    stb['label'].append(None)
                    stb['predict'].append(None)
                elif result_p is None:
                    prf_extensions = set()
                    for set1 in result_l:
                        if not any(set1[0].issubset(set2[0]) for set2 in result_l if set2 != set1):
                            prf_extensions.add(set1[0])
                    prf['label'].append(prf_extensions)
                    stb_extensions = set()
                    for set1 in result_l:
                        if (not set1[2]):
                            stb_extensions.add(set1[0])
                    stb['label'].append(stb_extensions)
                    prf['predict'].append(None)
                    stb['predict'].append(None)
                else:
                    # stb extension
                    stb_extensions = set()
                    for set1 in result_l:
                        if (not set1[2]):
                            stb_extensions.add(set1[0])
                    stb['label'].append(stb_extensions)
                    stb_extensions = set()
                    for set1 in result_p:
                        if (not set1[2]):
                            stb_extensions.add(set1[0])
                    stb['predict'].append(stb_extensions)

                    # prf extension
                    prf_extensions = set()
                    for set1 in result_l:
                        if not any(set1[0].issubset(set2[0]) for set2 in result_l if set2 != set1):
                            prf_extensions.add(set1[0])
                    prf['label'].append(prf_extensions)
                    prf_extensions = set()
                    for set1 in result_p:
                        if not any(set1[0].issubset(set2[0]) for set2 in result_p if set2 != set1):
                            prf_extensions.add(set1[0])
                    prf['predict'].append(prf_extensions)

    # 初始化统计数据
    grd_stats = {k: {'right': 0, 'total': 0} for k in range(26, 36)}
    com_stats = {k: {'right': 0, 'total': 0} for k in range(26, 36)}
    stb_stats = {k: {'right': 0, 'total': 0} for k in range(26, 36)}
    prf_stats = {k: {'right': 0, 'total': 0} for k in range(26, 36)}
    # 计算 grd 准确率
    for idx in range(len(grd['label'])):
        n_args = 26 + (idx // 100)  # 根据索引推断论证个数
        grd_stats[n_args]['total'] += 1
        if grd['label'][idx] is None:
            print(f"{idx} in is None : {grd['label'][idx]}")
        elif grd['predict'][idx] is not None and grd['label'][idx]['IN'] == grd['predict'][idx].get('IN', {}):
            grd_stats[n_args]['right'] += 1
            # print(f"[{idx}]grd_label:{grd['label'][idx]}\ngrd_predict:{grd['predict'][idx]}")
        else:
            print(f"[{idx}wrong]grd_label:{grd['label'][idx]}\ngrd_predict:{grd['predict'][idx]}")

    # 计算 com 准确率
    for idx in range(len(com['label'])):
        n_args = 26 + (idx // 100)  # 根据索引推断论证个数
        com_stats[n_args]['total'] += 1
        # print(idx)
        # print(com['label'][idx])
        if com['label'][idx] is None:
            print(f"{idx} in is None : {com['label'][idx]}")
        # if com['predict'][idx] is not None and com['label'][idx]['IN'] == com['predict'][idx].get('IN', {}):
        elif com['predict'][idx] is not None and set(com['label'][idx]) == set(com['predict'][idx]):
            com_stats[n_args]['right'] += 1
        # else:
        #     print(f"com_label:{com['label'][idx]}\ncom_predict:{com['predict'][idx]}")

    # 计算 stb 准确率
    for idx in range(len(stb['label'])):
        n_args = 26 + (idx // 100)  # 根据索引推断论证个数
        stb_stats[n_args]['total'] += 1
        # print(idx)
        if stb['label'][idx] is None:
            print(f"{idx} in is None : {stb['label'][idx]}")
        elif stb['predict'][idx] is not None and stb['label'][idx] == stb['predict'][idx]:
            stb_stats[n_args]['right'] += 1
        # else:
        #     print(f"stb_label:{stb['label'][idx]}\nstb_predict:{stb['predict'][idx]}")

    # 计算 prf 准确率
    for idx in range(len(prf['label'])):
        n_args = 26 + (idx // 100)  # 根据索引推断论证个数
        prf_stats[n_args]['total'] += 1
        # print(idx)
        if prf['label'][idx] is None:
            print(f"{idx} in is None : {prf['label'][idx]}")
        elif prf['predict'][idx] is not None and prf['label'][idx] == prf['predict'][idx]:
            prf_stats[n_args]['right'] += 1
        # else:
        #     print(f"prf_label:{prf['label'][idx]}\nprf_predict:{prf['predict'][idx]}")

    grd_total = 0
    grd_right = 0
    com_total = 0
    com_right = 0
    stb_total = 0
    stb_right = 0
    prf_total = 0
    prf_right = 0
    print(grd_stats)
    print(com_stats)
    print(stb_stats)
    print(prf_stats)
    # 打印结果
    for n_args in range(26, 36):
        if grd_stats[n_args]['total'] > 0:
            acc = grd_stats[n_args]['right'] / grd_stats[n_args]['total'] * 100
            print(f"grd {n_args} args accuracy: {acc:.2f}%")
            with open('qwen2_exp_scaling.txt', 'a') as f:
                print(f"grd {n_args} args accuracy: {acc:.2f}%", file=f)
            grd_total += grd_stats[n_args]['total']
            grd_right += grd_stats[n_args]['right']
        if com_stats[n_args]['total'] > 0:
            acc = com_stats[n_args]['right'] / com_stats[n_args]['total'] * 100
            print(f"com {n_args} args accuracy: {acc:.2f}%")
            com_total += com_stats[n_args]['total']
            com_right += com_stats[n_args]['right']
        if stb_stats[n_args]['total'] > 0:
            acc = stb_stats[n_args]['right'] / stb_stats[n_args]['total'] * 100
            print(f"stb {n_args} args accuracy: {acc:.2f}%")
            stb_total += stb_stats[n_args]['total']
            stb_right += stb_stats[n_args]['right']
        if prf_stats[n_args]['total'] > 0:
            acc = prf_stats[n_args]['right'] / prf_stats[n_args]['total'] * 100
            print(f"prf {n_args} args accuracy: {acc:.2f}%")
            prf_total += prf_stats[n_args]['total']
            prf_right += prf_stats[n_args]['right']
    # 计算并打印总体准确率
    print(file_path)
    grd_accuracy = grd_right / grd_total * 100
    print(f"Total grd accuracy: {grd_accuracy:.2f}%")
    with open('llama_grd_acc.txt', 'a') as f:
        print(f"{dir} grd accuracy: {grd_accuracy:.2f}%",file=f)
    com_accuracy = com_right / com_total * 100
    print(f"Total com accuracy: {com_accuracy:.2f}%")
    stb_accuracy = stb_right / stb_total * 100
    print(f"Total stb accuracy: {stb_accuracy:.2f}%")
    prf_accuracy = prf_right / prf_total * 100
    print(f"Total prf accuracy: {prf_accuracy:.2f}%")
