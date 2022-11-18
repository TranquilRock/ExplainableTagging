import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

CLS = 101
SEP = 102


class IRIEDataset(Dataset):
    def __init__(self, data, ids, mode):
        self.data = data
        self.ids = ids
        self.mode = mode
    
    def __len__(self) -> int:
        return len(self.ids)
        
    def __getitem__(self, index):
        ID = self.ids[index]
        q = self.data[ID][0]['q']
        r = self.data[ID][0]['r']
        
        if self.mode == 'train':
            q_plum = random.choice([d['q\''] for d in self.data[ID]])
            r_plum = random.choice([d['r\''] for d in self.data[ID]])
            return ID, torch.tensor(q['input_ids']), torch.tensor(r['input_ids']), torch.tensor(q_plum['input_ids']), torch.tensor(r_plum['input_ids'])
        elif self.mode == 'valid':
            q_plum = random.choice([d['q\''] for d in self.data[ID]])
            r_plum = random.choice([d['r\''] for d in self.data[ID]])
            return ID, q, r, q_plum, r_plum
        else:
            return ID, q, r
        
