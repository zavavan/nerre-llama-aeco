# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_doping_dataset,
    #get_dopingengextra_dataset,
    #get_dopingjson_dataset,
    get_genmat_dataset,
    get_AuNR_dataset,
    get_mof_dataset,
)
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "dopingjson_dataset": get_doping_dataset,
    "dopingengextra_dataset":get_doping_dataset,
    "dopingeng_dataset":get_doping_dataset,
#    "dopingjson_dataset": get_dopingjson_dataset,
#    "dopingengextra_dataset":get_dopingengextra_dataset,
    "generalmatfold0_dataset":get_genmat_dataset,
    "generalmatfold1_dataset":get_genmat_dataset,
    "generalmatfold2_dataset":get_genmat_dataset,
    "generalmatfold3_dataset":get_genmat_dataset,
    "generalmatfold4_dataset":get_genmat_dataset,
    "moffold0_dataset":get_mof_dataset,
    "moffold1_dataset":get_mof_dataset,
    "moffold2_dataset":get_mof_dataset,
    "moffold3_dataset":get_mof_dataset,
    "moffold4_dataset":get_mof_dataset,

    "AuNR_dataset": get_AuNR_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
