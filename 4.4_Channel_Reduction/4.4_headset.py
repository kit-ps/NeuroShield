import os
import json
import numpy as np
import h5py
import torch
from collections import defaultdict
import selfeeg

from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics.pairwise import cosine_distances as cd
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import cosine_similarity as cs
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity, SNRDistance
from pyeer.eer_info import get_eer_stats

# Ensure CUDA is visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===================================================================
# 1. Load full RAW EEG data
# ===================================================================
with h5py.File("../Data/test_raw.h5", "r") as f1, h5py.File("../Data/neg_raw.h5", "r") as f2:
    X1, Y1, S1, H1 = f1['data'][:], f1['labels'][:], f1['sessions'][:], f1['hardwares'][:]
    X2, Y2, S2, H2 = f2['data'][:], f2['labels'][:], f2['sessions'][:], f2['hardwares'][:]

X_all = np.concatenate([X1, X2], axis=0)
Y_all = np.concatenate([Y1, Y2], axis=0)
S_all = np.concatenate([S1, S2], axis=0)
H_all = np.concatenate([H1, H2], axis=0)

# ===================================================================
# 2. Headset configurations (only channel subsets!)
# ===================================================================
def get_headset_indices(headset):
    if headset == 'Emotive':
        return [6, 10, 15, 17, 21, 23, 27, 33, 37, 44, 58, 66, 79, 81]
    elif headset == 'DSIVR300':
        return [30, 60, 62, 64, 69, 77, 80]
    elif headset == 'Muse':
        return [4, 12, 46, 56]
    else:
        raise ValueError(f"Unknown headset: {headset}")

# ===================================================================
# 3. Model loading
# ===================================================================
def load_model_for_headset(headset):
    headset_indices = get_headset_indices(headset)
    emb, chans, samp = 256, len(headset_indices), 500
    state_path = f'../4.2_Analysis/Train_Models/model_3/ResNet1D_SupConLoss_{headset}_final.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = selfeeg.models.ResNet1D(nb_classes=emb, Chans=chans, Samples=samp).to(device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(state_path))
    model.eval()
    return model, device, headset_indices

# ===================================================================
# 4. Embedding extraction
# ===================================================================
def extract_embeddings(model, device, X_raw, headset_indices, batch_size=512):
    model.eval()
    X_raw = X_raw[:, headset_indices, :]
    embeddings = []
    with torch.no_grad():
        for i in range(0, X_raw.shape[0], batch_size):
            batch = torch.tensor(X_raw[i:i+batch_size], dtype=torch.float32).to(device)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

# ===================================================================
# 5. Unique subjects extraction
# ===================================================================
def get_unique_subjects(Y):
    return np.unique(Y)

# ===================================================================
# 6. Session splitting functions
# ===================================================================
def split_all_possible_verifications(X, Y, S, subjects, num_enrollment_sessions):
    e_X, e_Y, e_S, v_X, v_Y, v_S = [], [], [], [], [], []
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

def split_first_verification(X, Y, S, subjects, num_enrollment_sessions, num_verification_sessions):
    e_X, e_Y, e_S, v_X, v_Y, v_S = [], [], [], [], [], []
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

# ===================================================================
# 7. Scoring function
# ===================================================================
def calculate_similarity_scores_two2(e_emb, y_e, v_emb, y_v, s_v, V=3):
    similarity_results_by_class = []
    similarity_results_by_class_dict = defaultdict(list)
    unique_classes = np.unique(y_e)
    class_indices = [np.where(y_e == cls)[0] for cls in unique_classes]
    similarity_matrix = -1 * cd(v_emb, e_emb)

    for i in range(similarity_matrix.shape[0]):
        current_class = y_v[i]
        subject_session = s_v[i]
        current_ss = np.where((s_v == subject_session) & (y_v == current_class))[0]
        selected_indices = np.random.choice(current_ss, V, replace=True)
        selected_indices = np.append(selected_indices, i)
        predicted_scoresg = similarity_matrix[selected_indices]

        for cls in unique_classes:
            same_class_indices = class_indices[np.where(unique_classes == cls)[0][0]]
            maxscore = []
            for ir in range(V + 1):
                max_score = sum(sorted(predicted_scoresg[ir, same_class_indices], reverse=True)[:10]) / 10
                maxscore.append(max_score)
            max_score = sum(maxscore) / (V + 1)
            label = 1 if current_class == cls else 0
            similarity_results_by_class.append([max_score, label, current_class, cls, i, i])
            similarity_results_by_class_dict[cls].append([max_score, label, current_class, cls, i, cls])

    return similarity_results_by_class, similarity_results_by_class_dict

# ===================================================================
# 8. Evaluation functions
# ===================================================================
def EERf(results):
    results = np.array(results)
    genuine = results[results[:, 1] == 1][:, 0]
    impostor = results[results[:, 1] == 0][:, 0]
    stats_a = get_eer_stats(genuine, impostor)
    return stats_a.eer, stats_a.fmr100, stats_a.fmr1000, stats_a.fmr10000

def calculate_and_print_averages(y_train, results3):
    u, counts = np.unique(y_train, return_counts=True)
    eer_values, fmr100_values, fmr1000_values, fmr10000_values = [], [], [], []
    ii = 0

    for i in results3.keys():
        re = EERf(results3[i])
        eer, fmr100, fmr1000, fmr10000 = re
        eer_values.append(eer)
        fmr100_values.append(fmr100)
        fmr1000_values.append(fmr1000)
        fmr10000_values.append(fmr10000)
        ii += 1

    average_eer, std_eer = np.mean(eer_values) * 100, np.std(eer_values) * 100
    average_fmr100, std_fmr100 = np.mean(fmr100_values) * 100, np.std(fmr100_values) * 100
    average_fmr1000, std_fmr1000 = np.mean(fmr1000_values) * 100, np.std(fmr1000_values) * 100
    average_fmr10000, std_fmr10000 = np.mean(fmr10000_values) * 100, np.std(fmr10000_values) * 100

    print(f"Final Average EER: {average_eer:.4f} ± {std_eer:.4f}")
    print(f"Final Average FMR100: {average_fmr100:.4f} ± {std_fmr100:.4f}")
    print(f"Final Average FMR1000: {average_fmr1000:.4f} ± {std_fmr1000:.4f}")
    print(f"Final Average FMR10000: {average_fmr10000:.4f} ± {std_fmr10000:.4f}")

    return average_eer, std_eer, average_fmr100, std_fmr100, average_fmr1000, std_fmr1000, average_fmr10000, std_fmr10000

# ===================================================================
# 9. Saving function
# ===================================================================
def save_results_to_json(headset, experiment_name, avg_eer, std_eer):
    os.makedirs("experiment_results", exist_ok=True)
    filename = f"experiment_results/{headset}_{experiment_name}.json"
    data = {"Average_EER": avg_eer, "Std_EER": std_eer}
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# ===================================================================
# 10. Experiment runner
# ===================================================================
def run_experiment(X, Y, S, subjects, num_enroll, num_verif, V, mode, experiment_name=None, headset=None):
    if mode == 'baseline':
        e_X, e_Y, e_S, v_X, v_Y, v_S = split_all_possible_verifications(X, Y, S, subjects, num_enroll)
    else:
        e_X, e_Y, e_S, v_X, v_Y, v_S = split_first_verification(X, Y, S, subjects, num_enroll, num_verif)

    results_by_class, results_by_class_dict = calculate_similarity_scores_two2(e_X, e_Y, v_X, v_Y, v_S, V)
    results = calculate_and_print_averages(e_Y, results_by_class_dict)
    avg_eer, std_eer = results[0], results[1]

    if experiment_name and headset:
        save_results_to_json(headset, experiment_name, avg_eer, std_eer)

    return avg_eer

# ===================================================================
# 11. Full Table runner
# ===================================================================
def full_table_run(X, Y, S, subjects, headset):
    result = {}
    result['Baseline'] = run_experiment(X, Y, S, subjects, 1, 1, 0, 'baseline', 'Baseline', headset)
    result['4V'] = run_experiment(X, Y, S, subjects, 1, 1, 3, 'baseline', '4V', headset)
    result['E1 1V'] = run_experiment(X, Y, S, subjects, 1, 1, 0, 'first', 'E1_1V', headset)
    result['E1 4V'] = run_experiment(X, Y, S, subjects, 1, 1, 3, 'first', 'E1_4V', headset)
    result['E2 1V'] = run_experiment(X, Y, S, subjects, 2, 1, 0, 'first', 'E2_1V', headset)
    result['E2 4V'] = run_experiment(X, Y, S, subjects, 2, 1, 3, 'first', 'E2_4V', headset)
    return result

# ===================================================================
# 12. Master loop
# ===================================================================
headsets = ['Emotive', 'DSIVR300', 'Muse']

for headset in headsets:
    print(f"\nProcessing headset: {headset}")
    model, device, headset_indices = load_model_for_headset(headset)
    embeddings = extract_embeddings(model, device, X_all, headset_indices)
    subjects = get_unique_subjects(Y_all)
    table_results = full_table_run(embeddings, Y_all, S_all, subjects, headset)
    print(f"Results for {headset}: {table_results}")
