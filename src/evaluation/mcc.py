import json
from tqdm import tqdm
from pathlib import Path
from acc2 import extract_json, extract_com
from sklearn.metrics import matthews_corrcoef

root_dir = Path(__file__).parent
# 假设 lines 是已经加载的数据行列表
# 初始化数据结构
dirs = ["15428"]
qwen2_no_exp = ["15428"]
qwen2_exp = ["13474"]
llama3_exp_dirs = ["13472", "11789", "10105", "8421", "6737", "5052", "3368", "1684"]
llama3_no_exp_dirs = ["13496", "11812", "10125", "8437", "6750", "5062", "3375", "1687"]
# dirs = ["13472", "11789", "10105", "8421", "6737", "5052", "3368", "1684"]

for dir in dirs:
    # file_path = root_dir / "llama3" / "no_exp" / dir / 'generated_predictions.jsonl'
    # file_path = root_dir / "llama3" / "no_exp" / dir / 'generated_predictions.jsonl'
    file_path = root_dir / "qwen2" / "no_exp" / dir / 'generated_predictions.jsonl'
    print(file_path)
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
    grd_stats = {k: {'skep': 0, 'cred': 0, 'total': 0} for k in range(6, 26)}
    com_stats = {k: {'skep': 0, 'cred': 0, 'total': 0} for k in range(6, 26)}
    stb_stats = {k: {'skep': 0, 'cred': 0, 'total': 0} for k in range(6, 26)}
    prf_stats = {k: {'skep': 0, 'cred': 0, 'total': 0} for k in range(6, 26)}
    mcc_labels = [True, False]
    # 计算 grd 准确率
    for idx in range(len(grd['label'])):
        n_args = 6 + (idx // 100)  # 根据索引推断论证个数
        grd_stats[n_args]['total'] += 1
        if grd['predict'][idx] is not None and grd['label'][idx] is not None:
            set_arg = range(0, n_args)
            label_grd_arg = [a in grd['label'][idx]['IN'] for a in set_arg]
            predict_grd_arg = [a in grd['predict'][idx]['IN'] for a in set_arg]
            if label_grd_arg == predict_grd_arg:
                grd2 = 1
            else:
                grd2 = matthews_corrcoef(label_grd_arg, predict_grd_arg)

            grd_stats[n_args]['skep'] += grd2
            grd_stats[n_args]['cred'] += grd2

            # 计算 com 准确率
    for idx in range(len(com['label'])):
        n_args = 6 + (idx // 100)  # 根据索引推断论证个数
        com_stats[n_args]['total'] += 1
        # print(idx)
        com_label = com['label'][idx]
        com_predict = com['predict'][idx]

        if com_label is not None and com['predict'][idx] is not None:
            set_arg = range(0, n_args)
            label_com_arg = [all(a in s for s in set(com_label)) for a in set_arg]
            predict_com_arg = [all(a in s for s in set(com_predict)) for a in set_arg]
            label_com_arg2 = [any(a in s for s in set(com_label)) for a in set_arg]
            predict_com_arg2 = [any(a in s for s in set(com_predict)) for a in set_arg]
            if label_com_arg == predict_com_arg:
                # print(f"l:{label_com_arg},\np:{predict_com_arg}")
                com_skep = 1
            else:
                com_skep = matthews_corrcoef(label_com_arg, predict_com_arg)
            if label_com_arg2 == predict_com_arg2:
                # print(f"l:{label_com_arg2},\np:{predict_com_arg2}")
                com_cred = 1
            else:
                com_cred = matthews_corrcoef(label_com_arg2, predict_com_arg2)

            com_stats[n_args]['skep'] += com_skep
            com_stats[n_args]['cred'] += com_cred

    # 计算 stb 准确率
    for idx in range(len(stb['label'])):
        n_args = 6 + (idx // 100)  # 根据索引推断论证个数
        stb_stats[n_args]['total'] += 1
        # print(idx)
        stb_label = stb['label'][idx]
        stb_predict = stb['predict'][idx]
        if stb['predict'][idx] is not None and stb['label'][idx] is not None:
            set_arg = range(0, n_args)
            label_stb_arg = [all(a in s for s in set(stb_label)) for a in set_arg]
            predict_stb_arg = [all(a in s for s in set(stb_predict)) for a in set_arg]
            label_stb_arg2 = [any(a in s for s in set(stb_label)) for a in set_arg]
            predict_stb_arg2 = [any(a in s for s in set(stb_predict)) for a in set_arg]
            if label_stb_arg == predict_stb_arg:
                stb_skep = 1
            else:
                stb_skep = matthews_corrcoef(label_stb_arg, predict_stb_arg)
            if label_stb_arg2 == predict_stb_arg2:
                stb_cred = 1
            else:
                stb_cred = matthews_corrcoef(label_stb_arg2, predict_stb_arg2)

            stb_stats[n_args]['skep'] += stb_skep
            stb_stats[n_args]['cred'] += stb_cred

    # 计算 prf 准确率
    for idx in range(len(prf['label'])):
        n_args = 6 + (idx // 100)  # 根据索引推断论证个数
        prf_stats[n_args]['total'] += 1
        # print(idx)
        prf_label = prf['label'][idx]
        prf_predict = prf['predict'][idx]
        if prf['predict'][idx] is not None and prf['label'][idx] is not None:
            set_arg = range(0, n_args)
            label_prf_arg = [all(a in s for s in set(prf_label)) for a in set_arg]
            predict_prf_arg = [all(a in s for s in set(prf_predict)) for a in set_arg]
            label_prf_arg2 = [any(a in s for s in set(prf_label)) for a in set_arg]
            predict_prf_arg2 = [any(a in s for s in set(prf_predict)) for a in set_arg]
            if label_prf_arg == predict_prf_arg:
                prf_skep = 1
            else:
                prf_skep = matthews_corrcoef(label_prf_arg, predict_prf_arg)
            if label_prf_arg2 == predict_prf_arg2:
                prf_cred = 1
            else:
                prf_cred = matthews_corrcoef(label_prf_arg2, predict_prf_arg2)
            prf_stats[n_args]['skep'] += prf_skep
            prf_stats[n_args]['cred'] += prf_cred

    grd_total = 0
    grd_cred = 0
    grd_skep = 0
    com_total = 0
    com_cred = 0
    com_skep = 0
    stb_total = 0
    stb_cred = 0
    stb_skep = 0
    prf_total = 0
    prf_cred = 0
    prf_skep = 0
    print(grd_stats)
    print(com_stats)
    print(stb_stats)
    print(prf_stats)
    # 打印结果
    for n_args in range(6, 26):
        if grd_stats[n_args]['total'] > 0:
            skep = grd_stats[n_args]['skep'] / grd_stats[n_args]['total']
            cred = grd_stats[n_args]['cred'] / grd_stats[n_args]['total']
            print(f"grd {n_args} args skep_mcc: {skep:.4f}")
            print(f"grd {n_args} args cred_mcc: {cred:.4f}")
            grd_total += grd_stats[n_args]['total']
            grd_skep += grd_stats[n_args]['skep']
            grd_cred += grd_stats[n_args]['cred']
        if com_stats[n_args]['total'] > 0:
            skep = com_stats[n_args]['skep'] / com_stats[n_args]['total']
            cred = com_stats[n_args]['cred'] / com_stats[n_args]['total']
            print(f"com {n_args} args skep_mcc {skep:.4f}")
            print(f"com {n_args} args cred_mcc: {cred:.4f}")
            com_total += com_stats[n_args]['total']
            com_skep += com_stats[n_args]['skep']
            com_cred += com_stats[n_args]['cred']
        if stb_stats[n_args]['total'] > 0:
            skep = stb_stats[n_args]['skep'] / stb_stats[n_args]['total']
            cred = stb_stats[n_args]['cred'] / stb_stats[n_args]['total']
            print(f"stb {n_args} args skep_mcc: {skep:.4f}")
            print(f"stb {n_args} args cred_mcc: {cred:.4f}")
            stb_total += stb_stats[n_args]['total']
            stb_skep += stb_stats[n_args]['skep']
            stb_cred += stb_stats[n_args]['cred']
        if prf_stats[n_args]['total'] > 0:
            skep = prf_stats[n_args]['skep'] / prf_stats[n_args]['total']
            cred = prf_stats[n_args]['cred'] / prf_stats[n_args]['total']
            print(f"prf {n_args} args skep_mcc: {skep:.4f}")
            print(f"prf {n_args} args cred_mcc: {cred:.4f}")
            prf_total += prf_stats[n_args]['total']
            prf_skep += prf_stats[n_args]['skep']
            prf_cred += prf_stats[n_args]['cred']
    # 计算并打印总体准确率
    grd_skep_accuracy = grd_skep / grd_total
    print(f"Total grd skep_mcc: {grd_skep_accuracy:.4f}")
    com_skep_accuracy = com_skep / com_total
    com_cred_accuracy = com_cred / com_total
    print(f"Total com skep_mcc : {com_skep_accuracy:.4f}")
    print(f"Total com cred_mcc: {com_cred_accuracy:.4f}")
    stb_skep_accuracy = stb_skep / stb_total
    stb_cred_accuracy = stb_cred / stb_total
    print(f"Total stb skep_mcc : {stb_skep_accuracy:.4f}")
    print(f"Total stb cred_mcc: {stb_cred_accuracy:.4f}")
    prf_skep_accuracy = prf_skep / prf_total
    prf_cred_accuracy = prf_cred / prf_total
    print(f"Total prf skep_mcc : {prf_skep_accuracy:.4f}")
    print(f"Total prf cred_mcc: {prf_cred_accuracy:.4f}")
