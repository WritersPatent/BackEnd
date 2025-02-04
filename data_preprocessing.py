import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer
import random
from typing import List, Tuple, Optional

class TextAugmentation:
    def __init__(self, seed: int = 42):
        self.mecab = Mecab()
        random.seed(seed)
    
    def random_delete(self, text, p=0.1):
        words = self.mecab.morphs(text)
        if len(words) == 1:
            return text
        remaining = [word for word in words if random.random() > p]
        if len(remaining) == 0:
            return words[0]
        return ' '.join(remaining)
    
    def random_swap(self, text):
        words = self.mecab.morphs(text)
        if len(words) < 2:
            return text
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

class KoreanTextDataset(Dataset):
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int], 
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 is_training: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class KoreanGenerationDataset(Dataset):
    def __init__(self, 
                 texts: List[str], 
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }
