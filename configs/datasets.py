# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
NERRErepo_dir = '../NERRE/'

if NERRErepo_dir.endswith('/'):
    pass
else:
    NERRErepo_dir+='/'

@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class dopingjson_dataset:
    dataset: str = "dopingjson_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'doping/data/'+"doping_data_forllama_json"

@dataclass
class dopingengextra_dataset:
    dataset: str = "dopingengextra_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'doping/data/'+"doping_data_forllama_engextra"

@dataclass
class dopingeng_dataset:
    dataset: str = "dopingeng_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'doping/data/'+"doping_data_forllama_eng"

@dataclass
class generalmatfold0_dataset:
    dataset: str = "generalmatfold0_dataset"
    fold: str='fold_0'
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_general/fold_0"

@dataclass
class generalmatfold1_dataset:
    dataset: str = "generalmatfold1_dataset"
    fold: str='fold_1'
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_general/fold_1"

@dataclass
class generalmatfold2_dataset:
    dataset: str = "generalmatfold2_dataset"
    fold: str='fold_2'
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_general/fold_2"

@dataclass
class generalmatfold3_dataset:
    dataset: str = "generalmatfold3_dataset"
    fold: str='fold_3'
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_general/fold_3"

@dataclass
class generalmatfold4_dataset:
    dataset: str = "generalmatfold4_dataset"
    fold: str='fold_4'
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_general/fold_4"

@dataclass
class moffold0_dataset:
    dataset: str = "moffold0_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_mof_gpt3/fold_0"

@dataclass
class moffold1_dataset:
    dataset: str = "moffold1_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_mof_gpt3/fold_1"

@dataclass
class moffold2_dataset:
    dataset: str = "moffold2_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_mof_gpt3/fold_2"

@dataclass
class moffold3_dataset:
    dataset: str = "moffold3_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_mof_gpt3/fold_3"

@dataclass
class moffold4_dataset:
    dataset: str = "moffold4_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'general_and_mofs/data/'+"experiments_mof_gpt3/fold_4"


@dataclass
class scierc_aeco_json_dataset:
    dataset: str = "scierc_aeco_json_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = NERRErepo_dir+'scierc_aeco/data'

