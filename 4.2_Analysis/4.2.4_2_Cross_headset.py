import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances as ed, cosine_distances as cd
from sklearn.metrics.pairwise import cosine_similarity as cs, manhattan_distances as md
from pyeer.eer_info import get_eer_stats
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity, SNRDistance

# --------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# --------------------------------

# EER calculation functions
def EERf(genuine, impostor):
    genuine = np.array(genuine).ravel()
    impostor = np.array(impostor).ravel()
    stats_a = get_eer_stats(genuine, impostor)
    return stats_a.eer, stats_a.fmr100

def calculate_and_print_averages(genuine, impostor):
    eer_values = []
    for key in genuine.keys():
        re = EERf(genuine[key], impostor[key])  
        eer_values.append(re[0])  
    average_eer = np.mean(eer_values) * 100
    std_eer = np.std(eer_values) * 100
    return average_eer, std_eer

# Embedding function (identity since data is already embedded)
def compute_embedding_batch_two(x_test_batch, embedding_network, batch_size=100, device="cuda"):
    return x_test_batch

# Similarity calculation (full pairwise session combination)
def compute_similarity_per_pair(data):
    similarity_results = defaultdict(lambda: defaultdict(list))
    subjects = np.unique(data['Y'])

    for subject in subjects:
        indices = np.where(data['Y'] == subject)[0]
        subject_data = data['X'][indices]
        subject_sessions = data['S'][indices]
        subject_headsets = data['H'][indices]

        session_hardware_pairs = list(set(zip(subject_sessions, subject_headsets)))
        session_hardware_pairs.sort()

        for (sess1, hw1) in session_hardware_pairs:
            for (sess2, hw2) in session_hardware_pairs:
                if sess1 < sess2:
                    enrollment_indices = np.where((subject_sessions == sess1) & (subject_headsets == hw1))[0]
                    enrollment_data = subject_data[enrollment_indices]
                    if enrollment_data.ndim == 1:
                        enrollment_data = enrollment_data.reshape(1, -1)

                    verification_indices = np.where((subject_sessions == sess2) & (subject_headsets == hw2))[0]
                    for i2 in verification_indices:
                        verification_sample = subject_data[i2]
                        if verification_sample.ndim == 1:
                            verification_sample = verification_sample.reshape(1, -1)

                        distances = -ed(enrollment_data, verification_sample)
                        max_score = np.mean(sorted(distances, reverse=True)[:40])
                        hw_pair = tuple((hw1, hw2))
                        similarity_results[hw_pair][subject].append(float(max_score.item()))
    return similarity_results

# --------------------------------
# Load data
x_test_e = np.load('./files/x_test_e.npy')
y_test_e = np.load('./files/y_test_e.npy')
s_test_e = np.load('./files/s_test_e.npy')
h_test_e = np.load('./files/h_test_e.npy')

x_test_v = np.load('./files/x_test_v.npy')
y_test_v = np.load('./files/y_test_v.npy')
s_test_v = np.load('./files/s_test_v.npy')
h_test_v = np.load('./files/h_test_v.npy')

print(x_test_e.shape, y_test_e.shape, s_test_e.shape, h_test_e.shape)
print(x_test_v.shape, y_test_v.shape, s_test_v.shape, h_test_v.shape)

# Compute impostor scores (still using your assessment_model_data_two)
def calculate_similarity_scores_two(enrollment_embeddings, y_enrollment, verification_embeddings, y_verification, distance):
    similarity_results_by_class = []
    similarity_results_by_class_dict = defaultdict(list)
    unique_classes = np.unique(y_enrollment)
    class_indices = [np.where(y_enrollment == cls)[0] for cls in unique_classes]

    if distance == "cd":
        similarity_matrix = -cd(verification_embeddings, enrollment_embeddings)
    elif distance == "ed":
        similarity_matrix = -ed(verification_embeddings, enrollment_embeddings)

    for i in range(similarity_matrix.shape[0]):
        current_class = y_verification[i]
        predicted_scores = similarity_matrix[i]
        for cls in unique_classes:
            same_class_indices = class_indices[np.where(unique_classes == cls)[0][0]]
            max_score = np.mean(sorted(predicted_scores[same_class_indices], reverse=True)[:40])
            if current_class != cls:
                similarity_results_by_class.append([max_score, 0, current_class, cls, i])
                similarity_results_by_class_dict[cls].append([max_score])

    return similarity_results_by_class, similarity_results_by_class_dict

def assessment_model_data_two(enrollment_data, ye, verification_data, yv, e_network, distance):
    enrollment_embeddings = compute_embedding_batch_two(enrollment_data, e_network)
    verification_embeddings = compute_embedding_batch_two(verification_data, e_network)
    return calculate_similarity_scores_two(enrollment_embeddings, ye, verification_embeddings, yv, distance)

results2, impostor = assessment_model_data_two(x_test_e, y_test_e, x_test_v, y_test_v, None, distance='ed')

# Merge enrollment & verification into full dataset
x_test = np.concatenate((x_test_e, x_test_v), axis=0)
y_test = np.concatenate((y_test_e, y_test_v), axis=0)
s_test = np.concatenate((s_test_e, s_test_v), axis=0)
h_test = np.concatenate((h_test_e, h_test_v), axis=0)

data = {'X': x_test, 'Y': y_test, 'S': s_test, 'H': h_test}

# Compute full similarity results
similarity_results = compute_similarity_per_pair(data)

# Print pair-level results
for pair, subjects in similarity_results.items():
    print(f"Pair {pair}:")
    for subject, scores in subjects.items():
        avg_score = np.mean(scores)
        print(f"  Subject {subject}: Average Similarity = {avg_score:.4f}, Count = {len(scores)}")

# Compute hardware pair statistics
subject_data_dict = defaultdict(list)
for i in range(len(y_test)):
    subject_data_dict[y_test[i]].append((s_test[i], h_test[i]))

pair_results = defaultdict(lambda: {'count': 0, 'subjects': set()})
for subject, sess_hw_list in subject_data_dict.items():
    sess_hw_list_sorted = sorted(sess_hw_list)
    for i in range(len(sess_hw_list_sorted)):
        sess1, hw1 = sess_hw_list_sorted[i]
        for j in range(len(sess_hw_list_sorted)):
            sess2, hw2 = sess_hw_list_sorted[j]
            if sess1 < sess2:
                hw_pair = tuple((hw1, hw2))
                pair_results[hw_pair]['count'] += 1
                pair_results[hw_pair]['subjects'].add(subject)


print("========================== \n")

for pair, info in pair_results.items():
    print(f"Pair {pair}: {info['count']} times, Unique Subjects: {len(info['subjects'])}")


print("========================== \n")

# Global pair statistics from similarity_results
for pair, subjects in similarity_results.items():
    total_count = sum(len(scores) for scores in subjects.values())
    unique_subjects = len(subjects)
    print(f"Pair {pair}: {total_count} times, Unique Subjects: {unique_subjects}")


print("========================== \n")

# EER for selected pairs
print(calculate_and_print_averages(similarity_results[(b'Geodisi', b'HydroCe')], impostor))
print(calculate_and_print_averages(similarity_results[(b'HydroCe', b'Geodisi')], impostor))
print(calculate_and_print_averages(similarity_results[(b'BioSemi', b'HydroCe')], impostor))
print(calculate_and_print_averages(similarity_results[(b'HydroCe', b'BioSemi')], impostor))

