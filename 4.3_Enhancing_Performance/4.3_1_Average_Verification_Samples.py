import os
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances as cd
from pyeer.eer_info import get_eer_stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# -------------------------------------------------------------------
# EER calculation utilities
# -------------------------------------------------------------------
def EERf(results):
    results = np.array(results)
    genuine = results[results[:, 1] == 1][:, 0]
    impostor = results[results[:, 1] == 0][:, 0]
    stats = get_eer_stats(genuine, impostor)
    return stats.eer, stats.fmr100, stats.fmr1000, stats.fmr10000


def calculate_and_print_averages(y_train, results_dict):
    _, counts = np.unique(y_train, return_counts=True)

    eer_values, fmr100_values, fmr1000_values, fmr10000_values = [], [], [], []

    for idx, (key, results) in enumerate(results_dict.items()):
        eer, fmr100, fmr1000, fmr10000 = EERf(results)
        print(f"{key}: EER = {eer:.4f}, FMR100 = {fmr100:.4f}, FMR1000 = {fmr1000:.4f}, "
              f"FMR10000 = {fmr10000:.4f}, Count = {counts[idx]}")
        eer_values.append(eer)
        fmr100_values.append(fmr100)
        fmr1000_values.append(fmr1000)
        fmr10000_values.append(fmr10000)

    def report(name, values):
        avg = np.mean(values) * 100
        std = np.std(values) * 100
        print(f"Final Average {name}: {avg:.4f}")
        print(f"Final {name} Standard Deviation: {std:.4f}")
        print(f"${avg:.2f} \\pm {std:.2f}$")
        return avg, std

    avg_eer, std_eer = report("EER", eer_values)
    return avg_eer, std_eer


# -------------------------------------------------------------------
# Similarity computation utilities
# -------------------------------------------------------------------
def calculate_similarity_scores(enroll_emb, y_enroll, verify_emb, y_verify, s_verify, extra=3):
    similarity_results = []
    similarity_results_by_class = defaultdict(list)
    unique_classes = np.unique(y_enroll)
    class_indices = {cls: np.where(y_enroll == cls)[0] for cls in unique_classes}
    similarity_matrix = -cd(verify_emb, enroll_emb)

    for i in range(similarity_matrix.shape[0]):
        current_class = y_verify[i]
        session = s_verify[i]

        current_indices = np.where((s_verify == session) & (y_verify == current_class))[0]
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
# Embedding computation
# -------------------------------------------------------------------
def compute_embedding_batch(x_data, model, batch_size=150, device="cuda"):
    return x_data  # dummy pass-through for your current data


# -------------------------------------------------------------------
# Main evaluation pipeline
# -------------------------------------------------------------------
def run_assessment(enroll_data, y_enroll, verify_data, y_verify, model, s_verify, extra=1):
    enroll_emb = compute_embedding_batch(enroll_data, model)
    verify_emb = compute_embedding_batch(verify_data, model)
    return calculate_similarity_scores(enroll_emb, y_enroll, verify_emb, y_verify, s_verify, extra=extra)


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
# Run loop for multiple extra values and save to JSON
# -------------------------------------------------------------------
results_summary = {}

for extra_val in [0, 1, 3, 7, 15, 31]:
    print(f"\nRunning for extra = {extra_val}")
    results_all, results_by_class = run_assessment(
        x_test_e, y_test_e, x_test_v, y_test_v, model=None, s_verify=s_test_v, extra=extra_val
    )
    avg_eer, std_eer = calculate_and_print_averages(y_test_e, results_by_class)
    results_summary[extra_val] = {
        "EER": avg_eer,
        "EER_STD": std_eer
    }

# Save results into JSON
with open("eer_results.json", "w") as f:
    json.dump(results_summary, f, indent=4)

print("\nResults saved to eer_results.json")
