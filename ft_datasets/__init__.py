# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from .grammar_dataset import get_dataset as get_grammar_dataset
from .alpaca_dataset import InstructionDataset as get_alpaca_dataset
from .samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
#from .custom_dataset import DopedMatDatasetEngextra as get_dopingengextra_dataset
#from .custom_dataset import DopedMatDatasetJson as get_dopingjson_dataset
from .custom_dataset import DopedMatDataset as get_doping_dataset
from .custom_dataset import GenMatDataset as get_genmat_dataset
from .custom_dataset import MOFDataset as get_mof_dataset

from .custom_dataset import AuNRDataset as get_AuNR_dataset
from .custom_dataset import scierc_aeco_DatasetJson as get_scierc_aeco_json_dataset