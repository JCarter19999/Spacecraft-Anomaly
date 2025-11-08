from dataclasses import dataclass
import numpy as np
import torch, random

@dataclass
class Split:
    train: float
    val: float
    test: float

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def time_split(df, train=0.6, val=0.2):
    n = len(df)
    i = int(n*train)
    j = i + int(n*val)
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
