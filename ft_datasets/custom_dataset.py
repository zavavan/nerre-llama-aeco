import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset

class DopedMatDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=512):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"scheme_train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"scheme_val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


class DopedMatDatasetJson(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=512):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"jsonscheme_train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"jsonscheme_val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

class DopedMatDatasetEngextra(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=512):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"engextrascheme_train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"engextrascheme_val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

class DopedMatDatasetEng(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=512):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"engscheme_train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"engscheme_val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

class GenMatDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=620):
        #self.data = json.load(open(dataset_config.data_path))
        
        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"/train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"/val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)

        example = self.tokenizer.encode(example)
 #       print(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

class MOFDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=1024):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"/train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"/val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)

        example = self.tokenizer.encode(example)
#        print(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

class AuNRDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=1024):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"_train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"_val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#+'\n\n'#f"item['input']\n\n"
            
        example = prompt + item["output"]

        #print(example[:int(len(example)/2)])
        #print(example[int(len(example)/2):])
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


class scierc_aeco_DatasetJson(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=512):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            self.data = json.load(open(dataset_config.data_path+"train.json")) # self.data[0]["train"]  # Adjust this based on your dataset's structure
        else:
            self.data = json.load(open(dataset_config.data_path+"val.json"))# self.data[0]["validation"]  # Adjust this based on your dataset's structure

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }