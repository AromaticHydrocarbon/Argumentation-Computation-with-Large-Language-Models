# Argumentation Computation with Large Language Models : A Benchmark Study

---
This repository holds the code to produce data and experiments for using large language models(LLM) to determine the extensions of various abstract argumentation semantics
## Installation

---
Create a virtual python 3 environment (tested with 3.10) and install with pip or conda

### Pip
First install

* pytorch (tested with version 2.5.1)

with the cuda/cpu settings for your system, then install the other requirements with
    
`pip install -r requirements.txt`

### AF generators

Afterward go to the `src/data/generators/vendor` directory and compile the Argumentation Framework generators
with `./install.sh` (compiling requires Java, Ant and Maven).

## Generate Data

To generate Argumentation Frameworks we use [AFBenchGen2](https://sourceforge.net/projects/afbenchgen/)
, [AFGen](http://argumentationcompetition.org/2019/papers/ICCMA19_paper_3.pdf)
and [probo](https://sourceforge.net/projects/probo/) (thanks to the original authors!)

The `src/data` folder contains data generation scripts:

- `generate_apx.py` generates AFs in apx format.
- `apx_to_afs.py` converts apx files to ArgumentationFramework objects and computes extensions and argument acceptance
- `afs_to_enforcement` generates and solves status and extensions enforcement problems for an AF The apx files are

`./generate_data.sh`

## Experiments
Go to the `src/experiments/llama_factory` follow the README.md , install the Requirement

Download Qwen and Llama3 from  
https://huggingface.co/Qwen/Qwen2-7B-Instruct  
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Copy your train and test dataset to `src/experiments/llama_factory/data` and add the info to dataset_info
goto `examples/train_lora`fill the `llama3_lora_sft.yaml` and `qwen2_lora_sft.yaml` and train:  

    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml;
    llamafactory-cli train examples/train_lora/qwen2_lora_sft.yaml;

To predict, fill the `[llama3/qwen2]_lora_predict.yaml` and run:

    llamafactory-cli train examples/train_lora/llama3_lora_predict.yaml;
    llamafactory-cli train examples/train_lora/qwen2_lora_predict.yaml;

### Reference
This project uses the code from the following paper

[1] Craandijk, Dennis, and Floris Bex. "Enforcement heuristics for argumentation with deep reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 5. 2022.

[2] Zheng, Yaowei, et al. "Llamafactory: Unified efficient fine-tuning of 100+ language models." arXiv preprint arXiv:2403.13372 (2024).


