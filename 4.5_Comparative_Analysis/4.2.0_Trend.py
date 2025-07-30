import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import h5py
import numpy as np
import torch
import json
from utils_evaluation import (
    generate_embeddings,
    compute_similarity_scores,
    evaluate_eer_per_class,
    get_enrollment_verification_indices
)
import selfeeg
import torch.nn as nn

# ─── HEADSET FUNCTION ─────────────────────────────────────────────────────────
def get_headset_indices(headset):
    if headset == 'Emotive':
        return [6, 10, 15, 17, 21, 23, 27, 33, 37, 44, 58, 66, 79, 81]
    elif headset == 'DSIVR300':
        return [30, 60, 62, 64, 69, 77, 80]
    elif headset == 'Muse':
        return [4, 12, 46, 56]
    elif headset == 'All':
        return None
    else:
        raise ValueError(f"Unknown headset: {headset}")

# ─── LOAD FULL DATA ────────────────────────────────────────────────────────────
with h5py.File("../Data/test_raw.h5", "r") as f1, h5py.File("../Data/neg_raw.h5", "r") as f2:
    X1, Y1, S1, H1 = f1['data'][:], f1['labels'][:], f1['sessions'][:], f1['hardwares'][:]
    X2, Y2, S2, H2 = f2['data'][:], f2['labels'][:], f2['sessions'][:], f2['hardwares'][:]

X_all = np.concatenate([X1, X2], axis=0)
Y_all = np.concatenate([Y1, Y2], axis=0)
S_all = np.concatenate([S1, S2], axis=0)
H_all = np.concatenate([H1, H2], axis=0)

# ─── ENROLL/VERIFY SPLIT ───────────────────────────────────────────────────────
enroll_idxs, verify_idxs = get_enrollment_verification_indices(Y_all, S_all)
X_enroll_full, y_enroll = X_all[enroll_idxs], Y_all[enroll_idxs]
X_verify_full, y_verify = X_all[verify_idxs], Y_all[verify_idxs]

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
headsets = ['Emotive', 'DSIVR300', 'Muse', 'All']
train_numbers = [8, 16, 32, 64, 128, 230]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}

for headset in headsets:
    headset_indices = get_headset_indices(headset)
    chans = 93 if headset == 'All' else len(headset_indices)
    results[headset] = {}

    # Select channels
    if headset == 'All':
        X_enroll = X_enroll_full
        X_verify = X_verify_full
    else:
        X_enroll = X_enroll_full[:, headset_indices, :]
        X_verify = X_verify_full[:, headset_indices, :]

    for train_number in train_numbers:
        # Load model
        state_path = f'../4.2_Analysis/Train_Models/model_3/ResNet1D_SupConLoss_{headset}_{train_number}_final.pth'
        model = selfeeg.models.ResNet1D(nb_classes=256, Chans=chans, Samples=500).to(device)
        model = torch.compile(model)
        model.load_state_dict(torch.load(state_path))
        model.eval()

        # Extract embeddings
        en_emb = generate_embeddings(X_enroll, model, device=device)
        ve_emb = generate_embeddings(X_verify, model, device=device)

        # Evaluate EER
        sim_dict = compute_similarity_scores(en_emb, y_enroll, ve_emb, y_verify, distance='ed')
        avg_eer, std_eer = evaluate_eer_per_class(y_enroll, sim_dict)
        print(f"Headset: {headset}, TrainNum: {train_number}, Avg EER: {avg_eer:.4f}, Std: {std_eer:.4f}")

        results[headset][train_number] = {'avg_eer': avg_eer, 'std_eer': std_eer}

# ─── SAVE RESULTS TO JSON ─────────────────────────────────────────────────────
with open('eer_results.json', 'w') as f_json:
    json.dump(results, f_json, indent=4)
