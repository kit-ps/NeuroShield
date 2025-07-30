# python 4.2.0_Selected_Pipeline.py > 4.2.0_Selected_Pipeline.log 2>&1 &

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import h5py
import numpy as np
import torch
from utils_evaluation import (
    generate_embeddings,
    compute_similarity_scores,
    evaluate_eer_per_class,
    get_enrollment_verification_indices
)
import selfeeg
import torch.nn as nn

# ─── LOAD FULL DATA INTO NUMPY ───────────────────────────────────────────────
with h5py.File("../Data/test_raw.h5", "r") as f1, h5py.File("../Data/neg_raw.h5", "r") as f2:
    X1, Y1, S1, H1 = f1['data'][:], f1['labels'][:], f1['sessions'][:], f1['hardwares'][:]
    X2, Y2, S2, H2 = f2['data'][:], f2['labels'][:], f2['sessions'][:], f2['hardwares'][:]

X_all = np.concatenate([X1, X2], axis=0)
Y_all = np.concatenate([Y1, Y2], axis=0)
S_all = np.concatenate([S1, S2], axis=0)
H_all = np.concatenate([H1, H2], axis=0)

# ─── ENROLL/VERIFY SPLIT (Index-based) ───────────────────────────────────────
enroll_idxs, verify_idxs = get_enrollment_verification_indices(Y_all, S_all)

X_enroll = X_all[enroll_idxs]
y_enroll = Y_all[enroll_idxs]
X_verify = X_all[verify_idxs]
y_verify = Y_all[verify_idxs]

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
name, loss = 'ResNet1D', 'SupConLoss'
emb, chans, samp = 256, 93, 500
state_path = f'../4.2_Analysis/Train_Models/model_3/{name}_{loss}_final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = selfeeg.models.ResNet1D(nb_classes=emb, Chans=chans, Samples=samp).to(device)
model = torch.compile(model)
model.load_state_dict(torch.load(state_path))
model.eval()

# ─── EMBEDDING EXTRACTION ─────────────────────────────────────────────────────
en_emb = generate_embeddings(X_enroll, model, device=device)
ve_emb = generate_embeddings(X_verify, model, device=device)

# ─── EVALUATION ───────────────────────────────────────────────────────────────
sim_dict = compute_similarity_scores(en_emb, y_enroll, ve_emb, y_verify, distance='ed')
avg_eer, std_eer = evaluate_eer_per_class(y_enroll, sim_dict)
print(f"Avg EER: {avg_eer:.4f}, Std: {std_eer:.4f}")

# ─── SAVE EMBEDDINGS AND METADATA ────────────────────────────────────────────
os.makedirs('./files', exist_ok=True)

save_paths = {
    './files/x_test_e.npy': en_emb,
    './files/y_test_e.npy': y_enroll,
    './files/s_test_e.npy': S_all[enroll_idxs],
    './files/h_test_e.npy': H_all[enroll_idxs],
    './files/x_test_v.npy': ve_emb,
    './files/y_test_v.npy': y_verify,
    './files/s_test_v.npy': S_all[verify_idxs],
    './files/h_test_v.npy': H_all[verify_idxs],
}

for path, data in save_paths.items():
    if not os.path.exists(path):
        np.save(path, data)
    else:
        print(f"File {path} already exists. Skipping save.")
