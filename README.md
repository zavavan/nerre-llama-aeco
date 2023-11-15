# NERRE with Llama 2

### For the publication "*Structured information extraction from scientific text with large language models*" in Nature Communications by John Dagdelen*, Alexander Dunn*, Sanghoon Lee, Nicholas Walker, Andrew S. Rosen, Gerbrand Ceder, Kristin Persson, and Anubhav Jain.

If you are just looking to download the LoRA weights directly, use this url: 
[https://figshare.com/ndownloader/files/43044994](https://figshare.com/ndownloader/files/43044994) 
and view the data entry on [Figshare](https://figshare.com/articles/dataset/LoRA_weights_for_Llama-2_NERRE/24501331). For more details on how to use it see [Instructions](Instructions.md).  

# Llama-2 Fine-tuning / Inference for Information Extraction tasks in Materials Science

For the publication "Structured information extraction from scientific text with large language models" in Nature Communications by John Dagdelen*, Alexander Dunn*, Nicholas Walker, Sanghoon Lee, Andrew S. Rosen, Gerbrand Ceder, Kristin A. Persson, and Anubhav Jain.

This repository contains code for Llama-2 benchmark of [NERRE repo](https://github.com/lbnlp/NERRE). This repository is a fork of [facebookresearch/llama-recipes repo](https://github.com/facebookresearch/llama-recipes) and results can be reproduced using [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf) base model. Please refer to [the original repository's README](https://github.com/slee-lab/llama-recipes/blob/main/README.md) for requirements and installations, and also [license information](https://github.com/slee-lab/llama-recipes/blob/main/LICENSE).

### If you are just looking to download the weights and run inference with the models we have already fine tuned, read [Preparing Environment](#Preparing-Environment) and skip ahead to the inference section below.

# Preparing Environment

This work used installation environment and fine-tuning instructions described in [the original repo's README](README) on a single GPU (A100, 80GB memory). This repository used base model of quantized [Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf). Please note that you should after you would have to request and been granted access from Meta to use the Llama-2 base model.


# Llama-2-70B (8-bit) fine-tuning using LoRA on a single GPU

## Preparing Training Data

To reproduce fine-tuned model on doping task, first adjust the training data path in [datasets.py](configs/datasets.py) and [custom_dataset.py](ft_datasets/custom_dataset.py) to point to training data and test data in [NERRE doping repo](https://github.com/lbnlp/NERRE/tree/main/doping/data), [NERRE general and MOF repo](https://github.com/lbnlp/NERRE/tree/main/general_and_mofs/data). Note that the [custom_dataset.py](ft_datasets/custom_dataset.py) use the key 'input' and 'output' instead of 'prompt' and 'completion', respectively, so you should also adjust the keys in the training data. 

## Doping task

```bash
python llama_finetuning.py  \
  --use_peft \
  --peft_method lora \
  --quantization \
  --model_name '/path_of_model_folder/70B' \
  --output_dir 'path/of/saved/peft/model' \
  --batch_size_training 1 \
  --micro_batch_size 1 \
  --num_epochs 7 \
  --dataset dopingjson_dataset
```

For schemas besides `json`, use the datasets:
- `dopingengextra_dataset` for DopingExtra-English
- `dopingeng_dataset` for Doping-English

## General task

```bash
python llama_finetuning.py \
  --use_peft \
  --peft_method lora \
  --model_name '/path_of_model_folder/70B' 
  --output_dir 'path/of/saved/peft/model' \
  --quantization \
  --batch_size_training 1 \
  --micro_batch_size 1 \
  --num_epochs 4 \
  --dataset generalmatfold0_dataset
```

For cross validation folds besides fold 0 substitute 1, 2, 3, or 4 in place of `*` in the `--dataset generalmatfold*_dataset` argument.

## MOF task

```bash
python llama_finetuning.py \
  --use_peft \
  --peft_method lora \
  --model_name '/path_of_model_folder/70B' \
  --output_dir 'path/of/saved/peft/model' \
  --quantization \
  --batch_size_training 1 \
  --micro_batch_size 1 \
  --num_epochs 4 \
  --dataset moffold0_dataset
```

For cross validation folds besides fold 0 substitute 1, 2, 3, or 4 in place of `*` in the `--dataset moffold*_dataset` argument.

# Inference using fine-tuned Llama-2-70B (8-bit) on a single GPU

## Downloading weights

If you just want to use a fine-tuned model we show in the paper, first install the `requirements-nerre.txt` and then download the weights with the `download_nerre_weights.py` script provided in the root directory of this repo.

Alternatively, download the LoRA weights directly from this url: [https://figshare.com/ndownloader/files/43044994](https://figshare.com/ndownloader/files/43044994) and view the data entry on [Figshare](https://figshare.com/articles/dataset/LoRA_weights_for_Llama-2_NERRE/24501331).

```
$ pip install -r requirements-nerre.txt
$ python download_nerre_weights.py
```

The output will look like:

```
Downloading NERRE LoRA weights to /Users/ardunn/ardunn/lbl/nlp/ardunn_text_experiments/nerre_official_llama_supplementary_repo/lora_weights
/Users/ardunn/ardunn/lbl/nlp/ardunn_text_experiments/nerre_official_llama_supplementary_repo/lora_weights.tar.gz: 100%|██████████| 3.00G/3.00G [04:04<00:00, 13.2MiB/s]
MD5Sum was ec5dd3e51a8c176905775849410445dc
Weights downloaded, extracting to /Users/ardunn/ardunn/lbl/nlp/ardunn_text_experiments/nerre_official_llama_supplementary_repo/lora_weights...
Weights extracted to /Users/ardunn/ardunn/lbl/nlp/ardunn_text_experiments/nerre_official_llama_supplementary_repo/lora_weights...
```

The weights will be downloaded to `lora_weights` directory in the root directory of this repo. 
Then follow directions below to set the path to the exact model you would like to load.


## Doping task
For doping task, go to directory of [NERRE repo](https://github.com/lbnlp/NERRE/tree/main/doping) and use [step2_predict_llama2.py](step2_predict_llama2.py) instead of [step2_train_predict.py](https://github.com/lbnlp/NERRE/blob/main/doping/step2_train_predict.py) to make predictions of the test set.

```bash
export LLAMA2_70B_8bit=/PATH/TO/MODEL/70B_8bit/

python step2_train_predict.py predict \
  --inference_model_name='70b_8bit' \
  --lora_weights='path/of/saved/peft/model' \
  --inference_json_raw_output='/path/to/save/inferencefile' \
  --inference_json_final_output='/path/to/save/decodedfile' \
   --schema_type='json'

```

Where `path/of/saved/peft` model either points to the lora weights you downloaded or your own fine-tuned weights.
The `path/to/save/inferencefile` determines the path where the raw outputs (sequences) for the doping task will be saved.
The `path/to/save/decodedfile` determines the path where the "decoded" (i.e., in JSON format regardless of the LLM schema) is saved.

You can also substitute the `--schema_type` for `eng` or `engextra`. 

## General task

```bash
python generate_general_and_mof.py \
  --lora_weights 'path/of/saved/peft/model' \
  --results_dir '/path/to/save/inferencefile' \
  --task 'general' \
  --fold 0
```

Where `path/of/saved/peft` model either points to the lora weights you downloaded or your own fine-tuned weights. 

You can also substitute the `--fold` for 1, 2, 3, or 4. 

## MOF task

```bash
python generate_general_and_mof.py \
  --lora_weights Path/of/saved/PEFT/model \
  --results_dir '/path/to/save/inferencefile' \
  --task 'mof' \
  --fold 0
```

Where `path/of/saved/peft` model either points to the lora weights you downloaded or your own fine-tuned weights. 

You can also substitute the `--fold` for 1, 2, 3, or 4. 

# Evaluation of test set predictions

You can now go to the [NERRE repo](https://github.com/lbnlp/NERRE/tree/main) to evaluate the inference files for each task. Doping task uses [step3_score.py](https://github.com/lbnlp/NERRE/blob/main/doping/step3_score.py) and the other tasks use [results.py](https://github.com/lbnlp/NERRE/blob/main/general_and_mofs/results.py) to obtain scores.


# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)


---

## Original LLama 2 fine-tuning repo readme reproduced below.

---



# Llama 2 Fine-tuning / Inference Recipes and Examples

The 'llama-recipes' repository is a companion to the [Llama 2 model](https://github.com/facebookresearch/llama). The goal of this repository is to provide examples to quickly get started with fine-tuning for domain adaptation and how to run inference for the fine-tuned models. For ease of use, the examples use Hugging Face converted versions of the models. See steps for conversion of the model [here](#model-conversion-to-hugging-face).

Llama 2 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios. In order to help developers address these risks, we have created the [Responsible Use Guide](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf). More details can be found in our research paper as well. For downloading the models, follow the instructions on [Llama 2 repo](https://github.com/facebookresearch/llama).


# Table of Contents
1. [Quick start](#quick-start)
2. [Model Conversion](#model-conversion-to-hugging-face)
3. [Fine-tuning](#fine-tuning)
    - [Single GPU](#single-gpu)
    - [Multi GPU One Node](#multiple-gpus-one-node)
    - [Multi GPU Multi Node](#multi-gpu-multi-node)
4. [Inference](./docs/inference.md)
5. [Repository Organization](#repository-organization)
6. [License and Acceptable Use Policy](#license)



# Quick Start

[Llama 2 Jupyter Notebook](quickstart.ipynb): This jupyter notebook steps you through how to finetune a Llama 2 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum). The notebook uses parameter efficient finetuning (PEFT) and int8 quantization to finetune a 7B on a single GPU like an A10 with 24GB gpu memory.

**Note** All the setting defined in [config files](./configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.

**Note** In case need to run PEFT model with FSDP, please make sure to use the PyTorch Nightlies.

**For more in depth information checkout the following:**

* [Single GPU Fine-tuning](./docs/single_gpu.md)
* [Multi-GPU Fine-tuning](./docs/multi_gpu.md)
* [LLM Fine-tuning](./docs/LLM_finetuning.md)
* [Adding custom datasets](./docs/Dataset.md)
* [Inference](./docs/inference.md)
* [FAQs](./docs/FAQ.md)

## Requirements
To run the examples, make sure to install the requirements using

```bash
# python 3.9 or higher recommended
pip install -r requirements.txt

```

**Please note that the above requirements.txt will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

# Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Llama 2 model definition provided by Hugging Face's transformers library.

Given that the original checkpoint resides under models/7B you can install all requirements and convert the checkpoint with:

```bash
## Install HuggingFace Transformers from source
pip freeze | grep transformers ## verify it is version 4.31.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

# Fine-tuning

For fine-tuning Llama 2 models for your domain-specific use cases recipes for PEFT, FSDP, PEFT+FSDP have been included along with a few test datasets. For details see [LLM Fine-tuning](./docs/LLM_finetuning.md).

## Single and Multi GPU Finetune

If you want to dive right into single or multi GPU fine-tuning, run the examples below on a single GPU like A10, T4, V100, A100 etc.
All the parameters in the examples and recipes below need to be further tuned to have desired results based on the model, method, data and task at hand.

**Note:**
* To change the dataset in the commands below pass the `dataset` arg. Current options for dataset are `grammar_dataset`, `alpaca_dataset`and  `samsum_dataset`. A description of the datasets and how to add custom datasets can be found in [Dataset.md](./docs/Dataset.md). For  `grammar_dataset`, `alpaca_dataset` please make sure you use the suggested instructions from [here](./docs/single_gpu.md#how-to-run-with-different-datasets) to set them up.

* Default dataset and other LORA config has been set to `samsum_dataset`.

* Make sure to set the right path to the model in the [training config](./configs/training.py).

### Single GPU:

```bash
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

Here we make use of Parameter Efficient Methods (PEFT) as described in the next section. To run the command above make sure to pass the `peft_method` arg which can be set to `lora`, `llama_adapter` or `prefix`.

**Note** if you are running on a machine with multiple GPUs please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`

**Make sure you set [save_model](configs/training.py) in [training.py](configs/training.py) to save the model. Be sure to check the other training settings in [train config](configs/training.py) as well as others in the config folder as needed or they can be passed as args to the training script as well.**


### Multiple GPUs One Node:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that int8 quantization from bit&bytes currently is not supported in FSDP.

```bash

torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model

```

Here we use FSDP as discussed in the next section which can be used along with PEFT methods. To make use of PEFT methods with FSDP make sure to pass `use_peft` and `peft_method` args along with `enable_fsdp`. Here we are using `BF16` for training.

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up the fine-tuning job. This has been enabled in `optimum` library from HuggingFace as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model --use_fast_kernels
```

### Fine-tuning using FSDP Only

If you are interested in running full parameter fine-tuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --use_fast_kernels

```

### Fine-tuning using FSDP on 70B Model

If you are interested in running full parameter fine-tuning on the 70B model, you can enable `low_cpu_fsdp` mode as the following command. This option will load model on rank0 only before moving model to devices to construct FSDP. This can dramatically save cpu memory when loading large models like 70B (on a 8-gpu node, this reduces cpu memory from 2+T to 280G for 70B model). This has been tested with `BF16` on 16xA100, 80GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8 llama_finetuning.py --enable_fsdp --low_cpu_fsdp --pure_bf16 --model_name /patht_of_model_folder/70B --batch_size_training 1 --micro_batch_size 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

### Multi GPU Multi Node:

```bash

sbatch multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```
You can read more about our fine-tuning strategies [here](./docs/LLM_finetuning.md).


# Repository Organization
This repository is organized in the following way:

[configs](configs/): Contains the configuration files for PEFT methods, FSDP, Datasets.

[docs](docs/): Example recipes for single and multi-gpu fine-tuning recipes.

[ft_datasets](ft_datasets/): Contains individual scripts for each dataset to download and process. Note: Use of any of the datasets should be in compliance with the dataset's underlying licenses (including but not limited to non-commercial uses)


[inference](inference/): Includes examples for inference for the fine-tuned models and how to use them safely.

[model_checkpointing](model_checkpointing/): Contains FSDP checkpoint handlers.

[policies](policies/): Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode).

[utils](utils/): Utility files for:

- `train_utils.py` provides training/eval loop and more train utils.

- `dataset_utils.py` to get preprocessed datasets.

- `config_utils.py` to override the configs received from CLI.

- `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.

- `memory_utils.py` context manager to track different memory stats in train loop.

# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)
