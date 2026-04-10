import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from random import randrange
import random
from typing import Optional



___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


# --- NEW: generic, config-driven protocol parser (keeps your old helpers intact) ---
def _normalize_delim(d):
    """
    None -> split on any whitespace
    " "  -> treat like None (any whitespace), more robust than one literal space
    ","  -> split on comma
    any other string -> used as-is in str.split(d)
    """
    if d is None:
        return None
    if isinstance(d, str) and d.strip() == "":
        return None
    if d == " ":
        return None
    return d

def parse_protocol(path, *, delimiter: Optional[str], fileid_col: int, label_col: int, bonafide_label='bonafide', has_label: bool = True):
    """
    Generic protocol reader driven by config.
    Returns:
        if has_label: (labels_dict, key_list)
        else: key_list
    """
    delim = _normalize_delim(delimiter)
    keys = []
    labels = {}

    with open(path, 'r') as f:
        for i, raw in enumerate(f):
            if i == 0:
                continue  # skip header
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delim) if delim is not None else line.split()
            if fileid_col >= len(parts) or (has_label and label_col >= len(parts)):
                raise ValueError(f"Line has too few columns for configured indices: `{line}`")
            key = parts[fileid_col]
            if has_label:
                lab = parts[label_col]
                labels[key] = 1 if lab == bonafide_label else 0
            keys.append(key)

    return (labels, keys) if has_label else keys
# -----------------------------------------------------------------------------------
    

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
    

class Radar_Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, config):
        '''
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        '''
               
        self.list_IDs = list_IDs
        # self.labels = labels
        self.base_dir = base_dir
        self.config = config
        # self.cut=64600 # take ~4 sec audio (64600 samples)
        self.cut=config["data_config"]['num_samples'] # take 12 sec audio (192,000 samples)
            
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        utt_id = self.list_IDs[index]
        full_path = os.path.join(self.base_dir, utt_id + '.flac')
        
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Audio file missing: {full_path}")
        try:
            X, fs = librosa.load(full_path, sr=16000)
        except Exception as e:
            print(f"Warning: failed to load {full_path}: {e}")
            raise

        X_pad= pad(X,self.cut)
        x_inp= Tensor(X_pad)
        # target = self.labels[utt_id]
        
        return x_inp, utt_id