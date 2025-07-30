import os
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from pyeer.eer_info import get_eer_stats

# ------------------------------------------------------
# Load your data exactly as you have it:
# ------------------------------------------------------
x_test_e = np.load('../4.2_Analysis/files/x_test_e.npy')
y_test_e = np.load('../4.2_Analysis/files/y_test_e.npy')
s_test_e = np.load('../4.2_Analysis/files/s_test_e.npy')
h_test_e = np.load('../4.2_Analysis/files/h_test_e.npy')

x_test_v = np.load('../4.2_Analysis/files/x_test_v.npy')
y_test_v = np.load('../4.2_Analysis/files/y_test_v.npy')
s_test_v = np.load('../4.2_Analysis/files/s_test_v.npy')
h_test_v = np.load('../4.2_Analysis/files/h_test_v.npy')

# Combine enrollment + verification datasets
X_test = np.concatenate((x_test_e, x_test_v), axis=0)
Y_test = np.concatenate((y_test_e, y_test_v), axis=0)
S_test = np.concatenate((s_test_e, s_test_v), axis=0)

# ------------------------------------------------------
# Compute EER given similarity scores
# ------------------------------------------------------
def compute_eer(results):
    results = np.array(results)
    genuine = results[results[:, 1] == 1][:, 0]
    impostor = results[results[:, 1] == 0][:, 0]
    stats = get_eer_stats(genuine, impostor)
    return stats.eer

# ------------------------------------------------------
# Compute final average and std EER
# ------------------------------------------------------
def evaluate_eer(y_enrollment, results_by_class):
    subject_ids = np.unique(y_enrollment)
    eer_values = []
    for sid in subject_ids:
        subject_results = results_by_class[sid]
        eer = compute_eer(subject_results)
        eer_values.append(eer)
    avg_eer = np.mean(eer_values) * 100
    std_eer = np.std(eer_values) * 100
    return avg_eer, std_eer

# ------------------------------------------------------
# Similarity score function (your original logic simplified)
# ------------------------------------------------------
def calculate_similarity_scores(enrollment_embeddings, y_enrollment, verification_embeddings, y_verification, s_ver, extra=1, top_k=10):
    similarity_results_by_class = defaultdict(list)
    unique_classes = np.unique(y_enrollment)
    class_indices = {cls: np.where(y_enrollment == cls)[0] for cls in unique_classes}

    similarity_matrix = -cosine_distances(verification_embeddings, enrollment_embeddings)

    for i in range(similarity_matrix.shape[0]):
        current_class = y_verification[i]
        subject_session = s_ver[i]

        current_ss = np.where((s_ver == subject_session) & (y_verification == current_class))[0]
        selected_indices = np.append(np.random.choice(current_ss, extra, replace=True), i)
        selected_scores = similarity_matrix[selected_indices]

        for cls in unique_classes:
            indices_cls = class_indices[cls]
            max_scores = []
            for j in range(extra + 1):
                sorted_scores = np.sort(selected_scores[j, indices_cls])[::-1][:top_k]
                mean_score = np.mean(sorted_scores)
                max_scores.append(mean_score)
            final_score = np.mean(max_scores)
            label = 1 if current_class == cls else 0
            similarity_results_by_class[cls].append([final_score, label])

    return similarity_results_by_class

# ------------------------------------------------------
# Main assessment function
# ------------------------------------------------------
def run_experiment(enrollment_data, enrollment_labels, verification_data, verification_labels, s_ver, extra=1):
    enrollment_embeddings = enrollment_data  # Identity embedding
    verification_embeddings = verification_data
    similarity_results_by_class = calculate_similarity_scores(
        enrollment_embeddings, enrollment_labels,
        verification_embeddings, verification_labels, s_ver, extra=extra
    )
    avg_eer, std_eer = evaluate_eer(enrollment_labels, similarity_results_by_class)
    return avg_eer, std_eer

# ------------------------------------------------------
# Prepare enrollment/verification data for each ES/V pair
# ------------------------------------------------------
def prepare_data(X, Y, S, ES, V):
    subjects = np.unique(Y)
    enroll_x, enroll_y, enroll_s = [], [], []
    verify_x, verify_y, verify_s = [], [], []
    NAS = 0

    for subj in subjects:
        idx = np.where(Y == subj)[0]
        subj_sessions = S[idx]
        unique_sessions = np.unique(subj_sessions)

        if len(unique_sessions) < ES + V:
            continue

        NAS += 1
        enroll_sessions = unique_sessions[:ES]
        verify_sessions = unique_sessions[ES:ES + V]

        enroll_idx = idx[np.isin(subj_sessions, enroll_sessions)]
        verify_idx = idx[np.isin(subj_sessions, verify_sessions)]

        enroll_x.extend(X[enroll_idx])
        enroll_y.extend(Y[enroll_idx])
        enroll_s.extend(S[enroll_idx])

        verify_x.extend(X[verify_idx])
        verify_y.extend(Y[verify_idx])
        verify_s.extend(S[verify_idx])

    return (
        np.array(enroll_x), np.array(enroll_y), np.array(enroll_s),
        np.array(verify_x), np.array(verify_y), np.array(verify_s),
        NAS
    )

# ------------------------------------------------------
# Full Table VII reproduction loop
# ------------------------------------------------------
results_table = []

for ES in [1, 2, 3, 4, 5, 6]:
    for V in [0, 3]:
        enroll_x, enroll_y, enroll_s, verify_x, verify_y, verify_s, NAS = prepare_data(X_test, Y_test, S_test, ES, 1)

        if NAS == 0:
            print(f"ES={ES}, V={V}: No valid subjects")
            continue

        avg_eer, std_eer = run_experiment(enroll_x, enroll_y, verify_x, verify_y, verify_s, extra=V)

        results_table.append({
            "ES": ES,
            "V": V + 1,
            "NAS": NAS,
            "EER": round(avg_eer, 2),
            "STD_EER": round(std_eer, 2)
        })

        print(f"ES={ES}, V={V+1}, NAS={NAS}, EER={avg_eer:.2f}, STD={std_eer:.2f}")

# ------------------------------------------------------
# Save results to JSON file
# ------------------------------------------------------
with open("table_vii_results.json", "w") as f:
    json.dump(results_table, f, indent=4)

print("\nResults saved to table_vii_results.json")
