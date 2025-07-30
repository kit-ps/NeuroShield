import os
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances as cd
from pyeer.eer_info import get_eer_stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(42)  # for reproducibility

# -------------------------------------------------------------------
# EER calculation utilities (nan-safe)
# -------------------------------------------------------------------
def EERf(results):
    results = np.array(results)
    genuine = results[results[:, 1] == 1][:, 0]
    impostor = results[results[:, 1] == 0][:, 0]

    if len(genuine) == 0 or len(impostor) == 0:
        return np.nan, np.nan, np.nan, np.nan

    stats = get_eer_stats(genuine, impostor)
    return stats.eer, stats.fmr100, stats.fmr1000, stats.fmr10000

def calculate_and_print_averages(y_train, results_dict):
    _, counts = np.unique(y_train, return_counts=True)

    eer_values, fmr100_values, fmr1000_values, fmr10000_values = [], [], [], []

    for idx, (key, results) in enumerate(results_dict.items()):
        eer, fmr100, fmr1000, fmr10000 = EERf(results)
        if np.isnan(eer):  # skip classes without valid scores
            continue
        eer_values.append(eer)
        fmr100_values.append(fmr100)
        fmr1000_values.append(fmr1000)
        fmr10000_values.append(fmr10000)

    def report(name, values):
        avg = np.nanmean(values) * 100
        std = np.nanstd(values) * 100
        print(f"Final Average {name}: {avg:.4f}")
        print(f"Final {name} Standard Deviation: {std:.4f}")
        print(f"${avg:.2f} \\pm {std:.2f}$")
        return avg, std

    avg_eer, std_eer = report("EER", eer_values)
    avg_fmr100, std_fmr100 = report("FMR100", fmr100_values)
    avg_fmr1000, std_fmr1000 = report("FMR1000", fmr1000_values)
    avg_fmr10000, std_fmr10000 = report("FMR10000", fmr10000_values)

    return (avg_eer, std_eer, avg_fmr100, std_fmr100, avg_fmr1000, std_fmr1000, avg_fmr10000, std_fmr10000)

# -------------------------------------------------------------------
# Similarity computation utilities
# -------------------------------------------------------------------
def calculate_similarity_scores(enroll_emb, y_enroll, verify_emb, y_verify, s_verify, extra=1):
    similarity_results = []
    similarity_results_by_class = defaultdict(list)
    unique_classes = np.unique(y_enroll)
    class_indices = {cls: np.where(y_enroll == cls)[0] for cls in unique_classes}
    similarity_matrix = -cd(verify_emb, enroll_emb)

    for i in range(similarity_matrix.shape[0]):
        current_class = y_verify[i]
        session = s_verify[i]

        current_indices = np.where((s_verify == session) & (y_verify == current_class))[0]
        if len(current_indices) == 0:
            continue

        selected_indices = np.append(np.random.choice(current_indices, extra, replace=True), i)
        selected_scores = similarity_matrix[selected_indices]

        for cls in unique_classes:
            indices_cls = class_indices[cls]
            max_scores = [
                np.mean(sorted(selected_scores[j, indices_cls], reverse=True)[:40])
                for j in range(extra + 1)
            ]
            score = np.mean(max_scores)
            label = int(current_class == cls)
            entry = [score, label, current_class, cls, i, i]
            similarity_results.append(entry)
            similarity_results_by_class[cls].append(entry)

    return similarity_results, similarity_results_by_class

# -------------------------------------------------------------------
# Embedding computation (placeholder)
# -------------------------------------------------------------------
def compute_embedding_batch(x_data, model, batch_size=150, device="cuda"):
    return x_data  # pass-through; embeddings already provided

# -------------------------------------------------------------------
# Main evaluation pipeline
# -------------------------------------------------------------------
def run_assessment(enroll_data, y_enroll, verify_data, y_verify, model, s_verify, extra=1):
    enroll_emb = compute_embedding_batch(enroll_data, model)
    verify_emb = compute_embedding_batch(verify_data, model)
    return calculate_similarity_scores(enroll_emb, y_enroll, verify_emb, y_verify, s_verify, extra=extra)

# -------------------------------------------------------------------
# Subject-wise splitting functions
# -------------------------------------------------------------------
def split_all_possible_verifications(X, Y, S, num_enrollment_sessions):
    e_X, e_Y, e_S, v_X, v_Y, v_S = [], [], [], [], [], []
    subjects = np.unique(Y)
    for subject in subjects:
        subject_indices = np.where(Y == subject)[0]
        subject_sessions = S[subject_indices]
        unique_sessions = np.unique(subject_sessions)
        if len(unique_sessions) < (num_enrollment_sessions + 1):
            continue
        enroll_sessions = unique_sessions[:num_enrollment_sessions]
        verify_sessions = unique_sessions[num_enrollment_sessions:]
        e_X.extend(X[subject_indices][np.isin(subject_sessions, enroll_sessions)])
        e_Y.extend(Y[subject_indices][np.isin(subject_sessions, enroll_sessions)])
        e_S.extend(S[subject_indices][np.isin(subject_sessions, enroll_sessions)])
        v_X.extend(X[subject_indices][np.isin(subject_sessions, verify_sessions)])
        v_Y.extend(Y[subject_indices][np.isin(subject_sessions, verify_sessions)])
        v_S.extend(S[subject_indices][np.isin(subject_sessions, verify_sessions)])
    return np.array(e_X), np.array(e_Y), np.array(e_S), np.array(v_X), np.array(v_Y), np.array(v_S)

def split_first_verification(X, Y, S, num_enrollment_sessions, num_verification_sessions):
    e_X, e_Y, e_S, v_X, v_Y, v_S = [], [], [], [], [], []
    subjects = np.unique(Y)
    for subject in subjects:
        subject_indices = np.where(Y == subject)[0]
        subject_sessions = S[subject_indices]
        unique_sessions = np.unique(subject_sessions)
        if len(unique_sessions) < (num_enrollment_sessions + num_verification_sessions):
            continue
        enroll_sessions = unique_sessions[:num_enrollment_sessions]
        verify_sessions = unique_sessions[num_enrollment_sessions:num_enrollment_sessions + num_verification_sessions]
        e_X.extend(X[subject_indices][np.isin(subject_sessions, enroll_sessions)])
        e_Y.extend(Y[subject_indices][np.isin(subject_sessions, enroll_sessions)])
        e_S.extend(S[subject_indices][np.isin(subject_sessions, enroll_sessions)])
        v_X.extend(X[subject_indices][np.isin(subject_sessions, verify_sessions)])
        v_Y.extend(Y[subject_indices][np.isin(subject_sessions, verify_sessions)])
        v_S.extend(S[subject_indices][np.isin(subject_sessions, verify_sessions)])
    return np.array(e_X), np.array(e_Y), np.array(e_S), np.array(v_X), np.array(v_Y), np.array(v_S)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
x_test_e = np.load('../4.2_Analysis/files/x_test_e.npy')
y_test_e = np.load('../4.2_Analysis/files/y_test_e.npy')
s_test_e = np.load('../4.2_Analysis/files/s_test_e.npy')
h_test_e = np.load('../4.2_Analysis/files/h_test_e.npy')

x_test_v = np.load('../4.2_Analysis/files/x_test_v.npy')
y_test_v = np.load('../4.2_Analysis/files/y_test_v.npy')
s_test_v = np.load('../4.2_Analysis/files/s_test_v.npy')
h_test_v = np.load('../4.2_Analysis/files/h_test_v.npy')

# -------------------------------------------------------------------
# Merge enrollment and verification data for splitting
# -------------------------------------------------------------------
X_full = np.concatenate([x_test_e, x_test_v], axis=0)
Y_full = np.concatenate([y_test_e, y_test_v], axis=0)
S_full = np.concatenate([s_test_e, s_test_v], axis=0)

# -------------------------------------------------------------------
# Scenario definitions
# -------------------------------------------------------------------
scenarios = [
    {"name": "Scenario 1", "split": "all", "num_enroll": 1, "num_verify": None, "extra": 1},
    {"name": "Scenario 2", "split": "all", "num_enroll": 1, "num_verify": None, "extra": 4},
    {"name": "Scenario 3", "split": "first", "num_enroll": 1, "num_verify": 1, "extra": 1},
    {"name": "Scenario 4", "split": "first", "num_enroll": 1, "num_verify": 1, "extra": 4},
    {"name": "Scenario 5", "split": "first", "num_enroll": 2, "num_verify": 1, "extra": 1},
    {"name": "Scenario 6", "split": "first", "num_enroll": 2, "num_verify": 1, "extra": 4},
    {"name": "Scenario 7", "split": "first", "num_enroll": 2, "num_verify": 1, "extra": 16},
]

# -------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------
results_summary = {}

for scenario in scenarios:
    print(f"\nRunning {scenario['name']}")

    if scenario["split"] == "all":
        e_X, e_Y, e_S, v_X, v_Y, v_S = split_all_possible_verifications(
            X_full, Y_full, S_full, num_enrollment_sessions=scenario["num_enroll"]
        )
    else:
        e_X, e_Y, e_S, v_X, v_Y, v_S = split_first_verification(
            X_full, Y_full, S_full,
            num_enrollment_sessions=scenario["num_enroll"],
            num_verification_sessions=scenario["num_verify"]
        )

    results_all, results_by_class = run_assessment(
        e_X, e_Y, v_X, v_Y, model=None, s_verify=v_S, extra=scenario["extra"]-1
    )
    
    (avg_eer, std_eer, avg_fmr100, std_fmr100, avg_fmr1000, std_fmr1000, avg_fmr10000, std_fmr10000) = calculate_and_print_averages(e_Y, results_by_class)

    results_summary[scenario['name']] = {
        "EER": avg_eer,
        "EER_STD": std_eer,
        "FMR100": avg_fmr100,
        "FMR100_STD": std_fmr100,
        "FMR1000": avg_fmr1000,
        "FMR1000_STD": std_fmr1000,
        "FMR10000": avg_fmr10000,
        "FMR10000_STD": std_fmr10000
    }

# -------------------------------------------------------------------
# Save results into JSON
# -------------------------------------------------------------------
with open("eer_results_table.json", "w") as f:
    json.dump(results_summary, f, indent=4)

print("\nResults saved to eer_results_table.json")
