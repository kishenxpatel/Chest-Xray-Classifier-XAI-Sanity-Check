import torch
import random
import numpy as np

SEED = 42
RANDOM_SEEDS = [3, 12]
PATHOLOGIES = ['Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Atelectasis']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

