import os
import random
import copy
import json
import pickle
import time
from multiprocessing import Pool, cpu_count
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config
from src.data.projectgpt.openai_api import OpenAI_GPT4_mini_API

def process_dataset(dataset_name):
    af_dir = config.dataset_dir / dataset_name / "AFs"
    filenames = list(af_dir.glob("*.pkl"))
    json_dir = config.dataset_dir / dataset_name / "prompt" / "txt-gpt"
    grd_dir = json_dir / "prompt__grd"
    com_dir= json_dir / "prompt__com"
    json_dir.mkdir(parents=True, exist_ok=True)
    grds = list(grd_dir.glob("*.txt"))
    coms = list(com_dir.glob("*.txt"))
    grd_files = set(file.stem.replace('_grd', '') + ".pkl" for file in grds)
    com_files = set(file.stem.replace('_com', '') + ".pkl" for file in coms)
    filter_files = grd_files&com_files
    # 过滤掉在b_dir中有对应.txt文件的.pkl文件
    filtered_af_files = [file for file in filenames if file.name not in filter_files]
    print(len(filtered_af_files))

if __name__ == '__main__':
    t = time.time()
    random.seed(42)  # for reproducibility
    dataset_names = [f"train-{i}" for i in range(12, 20)]
    print(dataset_names)
    gpt = OpenAI_GPT4_mini_API()
    for dataset_name in dataset_names:
        process_dataset(dataset_name)