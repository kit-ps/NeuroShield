#python 4.2.4_1_Cross_Model.py > 4.2.4_1_Cross_Model.log 2>&1 &

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import json
from collections import defaultdict
from utils_evaluation import generate_embeddings, compute_similarity_scores, evaluate_eer_per_class

import selfeeg

# ===================== Model Definition =====================

def build_model(emb: int, chans: int, samp: int) -> nn.Module:
    return selfeeg.models.ResNet1D(nb_classes=emb, Chans=chans, Samples=samp)

# ===================== Enrollment/Verification Split =====================

def split_enrollment_verification(X, Y, S):
    subjects = np.unique(Y)
    X_enroll, Y_enroll = [], []
    X_verify, Y_verify = [], []

    for subject in subjects:
        indices = np.where(Y == subject)[0]
        subject_sessions = S[indices]

        if len(np.unique(subject_sessions)) < 2:
            continue

        min_session = np.min(subject_sessions)
        X_enroll.extend(X[indices][subject_sessions == min_session])
        Y_enroll.extend(Y[indices][subject_sessions == min_session])
        X_verify.extend(X[indices][subject_sessions != min_session])
        Y_verify.extend(Y[indices][subject_sessions != min_session])

    X_enroll, Y_enroll = np.array(X_enroll), np.array(Y_enroll)
    X_verify, Y_verify = np.array(X_verify), np.array(Y_verify)

    idx_e = np.random.permutation(len(X_enroll))
    idx_v = np.random.permutation(len(X_verify))
    X_enroll, Y_enroll = X_enroll[idx_e], Y_enroll[idx_e]
    X_verify, Y_verify = X_verify[idx_v], Y_verify[idx_v]

    return X_enroll, Y_enroll, X_verify, Y_verify

# ===================== Full Evaluation Pipeline =====================

def run_evaluation(model_path, dataset_files, device, model_config, distance="ed"):
    X, Y, S = [], [], []
    for file_path in dataset_files:
        with h5py.File(file_path, "r") as f:
            if 'X' in f:
                X.append(f['X'][:])
                Y.append(f['Y'][:])
                S.append(f['S'][:])
            else:
                X.append(f['data'][:])
                Y.append(f['labels'][:])
                S.append(f['sessions'][:])
    X, Y, S = np.concatenate(X), np.concatenate(Y), np.concatenate(S)

    idx = np.random.permutation(len(X))
    X, Y, S = X[idx], Y[idx], S[idx]

    X_enroll, Y_enroll, X_verify, Y_verify = split_enrollment_verification(X, Y, S)

    model = build_model(**model_config).to(device)
    model = torch.compile(model)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    emb_enroll = generate_embeddings(X_enroll, model, device=device)
    emb_verify = generate_embeddings(X_verify, model, device=device)


    similarity_results = compute_similarity_scores(emb_enroll, Y_enroll, emb_verify, Y_verify, distance)
    return evaluate_eer_per_class(Y_enroll, similarity_results)

# ===================== Entry Point =====================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_files = [
        ["../Data/test_raw.h5", "../Data/neg_raw.h5"],
        ["../Data/test_hardware_HydroCe.h5"],
        ["../Data/test_hardware_BioSemi.h5"],
        ["../Data/test_hardware_Geodisi.h5"]
    ]

    model_dir = "./Train_Models/model_3/"
    model_files = [
        "ResNet1D_SupConLoss_final.pth",
        "ResNet1D_SupConLoss_train_hardware_HydroCe_final.pth",
        "ResNet1D_SupConLoss_train_hardware_BioSemi_final.pth",
        "ResNet1D_SupConLoss_train_hardware_Geodisi_final.pth"
    ]

    model_config = {"emb": 256, "chans": 93, "samp": 500}

    results = []

    for model_name in model_files:
        model_path = os.path.join(model_dir, model_name)
        print(f"\nProcessing model: {model_path}")
        for files in dataset_files:
            print(f"  Dataset: {files}")
            avg_eer, std_eer = run_evaluation(model_path, files, device, model_config, distance="ed")

            results.append({
                "model": model_name,
                "dataset": [os.path.basename(f) for f in files],
                "avg_eer": round(avg_eer, 4),
                "std_eer": round(std_eer, 4)
            })
            print("model: ", model_name,
                "dataset: ", [os.path.basename(f) for f in files],
                "avg_eer: ", round(avg_eer, 4),
                "std_eer: ", round(std_eer, 4))

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to evaluation_results.json")
