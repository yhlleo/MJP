import torch
import torch.nn as nn

from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

import math
import numpy as np

from models.vit_timm import vit_small_patch16_224

model_path = "/path/to/model"
model_type = "deit"

configs = {
    'deit': {
        "use_unk": False,
        "use_idx_emb": False,
        "use_dlocr": False,
        "dlocr_type": None
    },
    "deit+js": {
        "use_unk": False,
        "use_idx_emb": False,
        "use_dlocr": False,
        "dlocr_type": None
    },
    "deit+js+unk": {
        "use_unk": True,
        "use_idx_emb": False,
        "use_dlocr": False,
        "dlocr_type": None
    },
    "deit+js+unk+idx": {
        "use_unk": True,
        "use_idx_emb": True,
        "use_dlocr": False,
        "dlocr_type": None
    },
    "deit+js+unk+dlocr(nln)": {
        "use_unk": True,
        "use_idx_emb": False,
        "use_dlocr": True,
        "dlocr_type": "nonlinear"
    },
    "deit+js+unk+dlocr(ln)": {
        "use_unk": True,
        "use_idx_emb": False,
        "use_dlocr": True,
        "dlocr_type": "linear"
    },
    "deit+js+unk+dlocr(pca)": {
        "use_unk": True,
        "use_idx_emb": False,
        "use_dlocr": True,
        "dlocr_type": "pca"
    },
}

def mse(model, x, y):
    return (np.abs((model.predict(x) - y))).mean()

def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
               (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def collect_pes(
    model_path, 
    model_type="deit", 
):
    checkpoint = torch.load(model_path, map_location="cpu")
    if 'unk' not in model_type:
        pe = checkpoint['model']['pos_embed'].squeeze(0)[1:]
    else:
        pe = checkpoint['model']['pos_embed.weight'][1:197]
    return pe.data.numpy()

pe = collect_pes(model_path, model_type)
y1 = np.random.permutation(pe.shape[0])
y2 = np.random.permutation(pe.shape[0])
x1 = pe[y1]
x2 = pe[y2]
model = LinearRegression()
print("1D:", cross_val_score(model, x1-x2, np.abs(y1-y2), cv=5, scoring=mse).mean())

y = np.arange(196).astype("uint8")
y = np.stack([y//14, y%14], axis=1)
rnd1 = np.random.permutation(pe.shape[0])
rnd2 = np.random.permutation(pe.shape[0])
x1, y1 = pe[rnd1], y[rnd1]
x2, y2 = pe[rnd2], y[rnd2]
model = LinearRegression()
print("2D:", cross_val_score(model, x1-x2, np.abs(y1-y2), cv=5, scoring=mse).mean())







