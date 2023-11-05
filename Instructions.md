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
