# NERRE with Llama 2

### For the publication "*Structured information extraction from scientific text with large language models*" in Nature Communications by John Dagdelen*, Alexander Dunn*, Sanghoon Lee, Nicholas Walker, Andrew S. Rosen, Gerbrand Ceder, Kristin Persson, and Anubhav Jain.

If you are just looking to download the LoRA weights directly, use this url: 
[https://figshare.com/ndownloader/files/43044994](https://figshare.com/ndownloader/files/43044994) 
and view the data entry on [Figshare](https://figshare.com/articles/dataset/LoRA_weights_for_Llama-2_NERRE/24501331).

# Llama-2 Fine-tuning / Inference for Information Extraction tasks in Materials Science

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
export LLAMA2_70B_8bit='meta-llama/Llama-2-70b-hf'

python step2_predict_llama2.py predict \
  --inference_model_name='70b_8bit' \
  --lora_weights='../../nerre-llama/lora_weights/llama2_70b_8bit_doping_json_7epoch' \
  --schema_type='json'

```
LLAMA2_70B_8bit is either a path of downloaded Llama-2-70b weights or [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) as shown in this example command;
lora_weights either points to the path where lora weights is downloaded or your own fine-tuned weights;

You can also substitute the `--schema_type` for `eng` or `engextra`. Please make sure to change `lora_weights` accordingly (e.g. lora_weights/llama2_70b_8bit_doping_eng_7epoch).

We run this on a single A100 (80GB VRAM) with CUDA runtime v11.7. The output will look like below but it is ok to ignore lines regarding training JSONL, as this code is only used for predicting with Llama-2:
```
===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/.conda/envs/nerrellamagittest/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda117.so
CUDA SETUP: CUDA runtime path found: /opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64/libcudart.so.11.0
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /home/.conda/envs/nerrellamagittest/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda117.so...
Doing 'predict' operation with schema type 'json'.
Using training json of /home/llamagittest/NERRE/doping/data/train.json, saving formatted output file to None.
Training JSONL file will be saved to /home/llamagittest/NERRE/doping/data/training_json_2023-XX-XX_XX.XX.XX.jsonl
Inference JSONL file will be saved to /home/llamagittest/NERRE/doping/data/inference_raw_json_2023-XX-XX_XX.XX.XX.json
Loaded 31 samples for inference.


70b_8bit


^MLoading checkpoint shards:   0%|          | 0/15 [00:00<?, ?it/s]^MLoading checkpoint shards:   7%|▋         | 1/15 [00:06<01:30,  6.50s/it]^MLoading checkpoint shards:  13%|█▎        | 2/15 [00:13<01:24,  6.52s/it]^MLoading checkpoint shards:  20%|██        | 3/15 [00:19<01:18,  6.53s/it]^MLoading checkpoint shards:  27%|██▋       | 4/15 [00:26<01:12,  6.55s/it]^MLoading checkpoint shards:  33%|███▎      | 5/15 [00:32<01:05,  6.57s/it]^MLoading checkpoint shards:  40%|████      | 6/15 [00:39<00:58,  6.53s/it]^MLoading checkpoint shards:  47%|████▋     | 7/15 [00:45<00:52,  6.55s/it]^MLoading checkpoint shards:  53%|█████▎    | 8/15 [00:52<00:46,  6.57s/it]^MLoading checkpoint shards:  60%|██████    | 9/15 [00:58<00:39,  6.55s/it]^MLoading checkpoint shards:  67%|██████▋   | 10/15 [01:05<00:32,  6.54s/it]^MLoading checkpoint shards:  73%|███████▎  | 11/15 [01:12<00:26,  6.58s/it]^MLoading checkpoint shards:  80%|████████  | 12/15 [01:18<00:19,  6.57s/it]^MLoading checkpoint shards:  87%|████████▋ | 13/15 [01:25<00:13,  6.56s/it]^MLoading checkpoint shards:  93%|█████████▎| 14/15 [01:32<00:06,  6.80s/it]^MLoading checkpoint shards: 100%|██████████| 15/15 [01:33<00:00,  4.92s/it]^MLoading checkpoint shards: 100%|██████████| 15/15 [01:33<00:00,  6.21s/it]
^MTexts processed:   0%|          | 0/31 [00:00<?, ?it/s]^MTexts processed:   3%|▎         | 1/31 [03:55<1:57:32, 235.10s/it]^MTexts processed:   6%|▋         | 2/31 [04:22<54:39, 113.09s/it]  ^MTexts processed:  10%|▉         | 3/31 [05:03<37:17, 79.93s/it] ^MTexts processed:  13%|█▎        | 4/31 [05:57<31:24, 69.80s/it]^MTexts processed:  16%|█▌        | 5/31 [06:48<27:22, 63.19s/it]^MTexts processed:  19%|█▉        | 6/31 [07:05<19:46, 47.45s/it]^MTexts processed:  23%|██▎       | 7/31 [08:32<24:06, 60.27s/it]^MTexts processed:  26%|██▌       | 8/31 [08:48<17:38, 46.01s/it]^MTexts processed:  29%|██▉       | 9/31 [09:40<17:38, 48.10s/it]^MTexts processed:  32%|███▏      | 10/31 [10:05<14:18, 40.87s/it]^MTexts processed:  35%|███▌      | 11/31 [11:09<15:59, 47.96s/it]^MTexts processed:  39%|███▊      | 12/31 [12:31<18:30, 58.44s/it]^MTexts processed:  42%|████▏     | 13/31 [12:52<14:07, 47.10s/it]^MTexts processed:  45%|████▌     | 14/31 [13:03<10:11, 36.00s/it]^MTexts processed:  48%|████▊     | 15/31 [13:56<10:58, 41.16s/it]^MTexts processed:  52%|█████▏    | 16/31 [14:23<09:13, 36.88s/it]^MTexts processed:  55%|█████▍    | 17/31 [17:13<17:57, 76.97s/it]^MTexts processed:  58%|█████▊    | 18/31 [19:30<20:35, 95.05s/it]^MTexts processed:  61%|██████▏   | 19/31 [20:08<15:34, 77.90s/it]^MTexts processed:  65%|██████▍   | 20/31 [20:41<11:50, 64.55s/it]^MTexts processed:  68%|██████▊   | 21/31 [21:41<10:29, 62.96s/it]^MTexts processed:  71%|███████   | 22/31 [23:40<11:59, 79.95s/it]^MTexts processed:  74%|███████▍  | 23/31 [24:15<08:51, 66.39s/it]^MTexts processed:  77%|███████▋  | 24/31 [25:37<08:17, 71.05s/it]^MTexts processed:  81%|████████  | 25/31 [26:03<05:44, 57.48s/it]^MTexts processed:  84%|████████▍ | 26/31 [28:53<07:36, 91.28s/it]^MTexts processed:  87%|████████▋ | 27/31 [29:11<04:37, 69.31s/it]^MTexts processed:  90%|█████████ | 28/31 [29:45<02:55, 58.59s/it]^MTexts processed:  94%|█████████▎| 29/31 [31:17<02:17, 68.61s/it]^MTexts processed:  97%|█████████▋| 30/31 [31:32<00:52, 52.66s/it]^MTexts processed: 100%|██████████| 31/31 [32:09<00:00, 47.93s/it]^MTexts processed: 100%|██████████| 31/31 [32:09<00:00, 62.24s/it]
Dumped 31 total to /home/llamagittest/NERRE/doping/data/inference_raw_json_2023-XX-XX_XX.XX.XX.json (and raw jsonl to /home/llamagittest/NERRE/doping/data/inference_raw_json_2023-XX-XX_XX.XX.XX.jsonl).
^M  0%|          | 0/31 [00:00<?, ?it/s]^M100%|██████████| 31/31 [00:00<00:00, 81112.55it/s]
Decoded 31 samples to file /home/llamagittest/NERRE/doping/data/inference_decoded_json_2023-XX-XX_XX.XX.XX.json
Time 0:33:47.900304

```

## General task

```bash
python generate_general_and_mof.py \
  --base_model 'meta-llama/Llama-2-70b-hf' \
  --lora_weights './lora_weights/llama2_70b_8bit_genmat_fold0_4epoch' \
  --test_data_path '../NERRE/general_and_mofs/data/experiments_general/fold_0/val.jsonl' \
  --results_dir './results_general_fold0' \
  --fold 0
```
base_model is a path of Llama-2-70b or [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) as shown in this example command;
lora_weights either points to the lora weights you downloaded or your own fine-tuned weights;
test_data_path either points to test data to run inference on (in NERRE repo for this example) or your own prompts to run inference on (Note that this is defaulted to a jsonl file each having text under 'prompt' key;
results_dir points to a directory to save the predictions.

You can also substitute the `--fold` for 1, 2, 3, or 4 to reproduce the test data inferences.

## MOF task

```bash
python generate_general_and_mof.py \
  --base_model 'meta-llama/Llama-2-70b-hf' \
  --lora_weights './lora_weights/llama2_70b_8bit_mof_fold0_4epoch' \
  --test_data_path '../NERRE/general_and_mofs/data/experiments_mof_gpt3/fold_0/val.jsonl' \
  --results_dir './results_mof_fold0' \
  --fold 0
```

Where arguments of this is the same as general task.

We run this on a single A100 (80GB VRAM) with CUDA runtime v11.7. The output will look like:
```
/home/.conda/envs/nerrellamagittest/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /home/.conda/envs/nerrellamagittest did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/.conda/envs/nerrellamagittest/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda117.so
CUDA SETUP: CUDA runtime path found: /opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /home/.conda/envs/nerrellamagittest/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda117.so...
^MLoading checkpoint shards:   0%|          | 0/15 [00:00<?, ?it/s]^MLoading checkpoint shards:   7%|▋         | 1/15 [00:05<01:11,  5.10s/it]^MLoading checkpoint shards:  13%|█▎        | 2/15 [00:10<01:05,  5.07s/it]^MLoading checkpoint shards:  20%|██        | 3/15 [00:15<01:01,  5.11s/it]^MLoading checkpoint shards:  27%|██▋       | 4/15 [00:20<00:56,  5.10s/it]^MLoading checkpoint shards:  33%|███▎      | 5/15 [00:25<00:50,  5.06s/it]^MLoading checkpoint shards:  40%|████      | 6/15 [00:30<00:45,  5.04s/it]^MLoading checkpoint shards:  47%|████▋     | 7/15 [00:35<00:40,  5.06s/it]^MLoading checkpoint shards:  53%|█████▎    | 8/15 [00:40<00:35,  5.07s/it]^MLoading checkpoint shards:  60%|██████    | 9/15 [00:45<00:30,  5.06s/it]^MLoading checkpoint shards:  67%|██████▋   | 10/15 [00:50<00:25,  5.05s/it]^MLoading checkpoint shards:  73%|███████▎  | 11/15 [00:55<00:20,  5.06s/it]^MLoading checkpoint shards:  80%|████████  | 12/15 [01:00<00:15,  5.06s/it]^MLoading checkpoint shards:  87%|████████▋ | 13/15 [01:05<00:10,  5.04s/it]^MLoading checkpoint shards:  93%|█████████▎| 14/15 [01:13<00:05,  5.85s/it]^MLoading checkpoint shards: 100%|██████████| 15/15 [01:13<00:00,  4.18s/it]^MLoading checkpoint shards: 100%|██████████| 15/15 [01:13<00:00,  4.92s/it]
^M  0%|          | 0/51 [00:00<?, ?it/s]^M  2%|▏         | 1/51 [00:50<41:41, 50.04s/it]^M  4%|▍         | 2/51 [01:12<27:24, 33.56s/it]^M  6%|▌         | 3/51 [01:33<22:20, 27.94s/it]^M  8%|▊         | 4/51 [02:07<23:51, 30.45s/it]^M 10%|▉         | 5/51 [02:32<21:54, 28.58s/it]^M 12%|█▏        | 6/51 [03:19<25:58, 34.62s/it]^M 14%|█▎        | 7/51 [03:42<22:37, 30.85s/it]^M 16%|█▌        | 8/51 [06:24<52:10, 72.79s/it]^M 18%|█▊        | 9/51 [08:07<57:32, 82.19s/it]^M 20%|█▉        | 10/51 [08:57<49:14, 72.06s/it]^M 22%|██▏       | 11/51 [09:31<40:25, 60.63s/it]^M 24%|██▎       | 12/51 [09:55<32:03, 49.32s/it]^M 25%|██▌       | 13/51 [10:38<30:06, 47.54s/it]^M 27%|██▋       | 14/51 [11:23<28:47, 46.68s/it]^M 29%|██▉       | 15/51 [12:22<30:13, 50.37s/it]^M 31%|███▏      | 16/51 [12:44<24:29, 41.99s/it]^M 33%|███▎      | 17/51 [13:08<20:36, 36.37s/it]^M 35%|███▌      | 18/51 [13:40<19:19, 35.13s/it]^M 37%|███▋      | 19/51 [14:02<16:34, 31.09s/it]^M 39%|███▉      | 20/51 [14:36<16:31, 31.99s/it]^M 41%|████      | 21/51 [15:44<21:29, 42.97s/it]^M 43%|████▎     | 22/51 [16:06<17:39, 36.52s/it]^M 45%|████▌     | 23/51 [16:28<15:04, 32.30s/it]^M 47%|████▋     | 24/51 [17:11<15:54, 35.34s/it]^M 49%|████▉     | 25/51 [18:41<22:25, 51.74s/it]^M 51%|█████     | 26/51 [20:07<25:56, 62.26s/it]^M 53%|█████▎    | 27/51 [20:33<20:30, 51.27s/it]^M 55%|█████▍    | 28/51 [21:06<17:33, 45.82s/it]^M 57%|█████▋    | 29/51 [22:12<19:02, 51.91s/it]^M 59%|█████▉    | 30/51 [23:01<17:50, 50.96s/it]^M 61%|██████    | 31/51 [23:27<14:26, 43.33s/it]^M 63%|██████▎   | 32/51 [25:29<21:12, 66.99s/it]^M 65%|██████▍   | 33/51 [25:59<16:50, 56.11s/it]^M 67%|██████▋   | 34/51 [26:24<13:14, 46.75s/it]^M 69%|██████▊   | 35/51 [26:44<10:19, 38.73s/it]^M 71%|███████   | 36/51 [28:42<15:33, 62.26s/it]^M 73%|███████▎  | 37/51 [29:04<11:46, 50.43s/it]^M 75%|███████▍  | 38/51 [29:34<09:33, 44.13s/it]^M 76%|███████▋  | 39/51 [30:18<08:50, 44.21s/it]^M 78%|███████▊  | 40/51 [31:13<08:42, 47.48s/it]^M 80%|████████  | 41/51 [31:39<06:50, 41.07s/it]^M 82%|████████▏ | 42/51 [32:10<05:42, 38.01s/it]^M 84%|████████▍ | 43/51 [32:28<04:16, 32.02s/it]^M 86%|████████▋ | 44/51 [32:51<03:23, 29.07s/it]^M 88%|████████▊ | 45/51 [33:15<02:45, 27.63s/it]^M 90%|█████████ | 46/51 [33:37<02:10, 26.04s/it]^M 92%|█████████▏| 47/51 [35:26<03:23, 50.93s/it]^M 94%|█████████▍| 48/51 [35:59<02:16, 45.40s/it]^M 96%|█████████▌| 49/51 [36:26<01:20, 40.10s/it]^M 98%|█████████▊| 50/51 [37:17<00:43, 43.31s/it]^M100%|██████████| 51/51 [38:04<00:00, 44.45s/it]^M100%|██████████| 51/51 [38:04<00:00, 44.80s/it]
Time consumed  0:39:22.807114
```

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
