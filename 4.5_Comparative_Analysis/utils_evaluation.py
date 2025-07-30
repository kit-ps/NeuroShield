import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances as ed, cosine_distances as cd
from pyeer.eer_info import get_eer_stats
import torch
from sklearn.preprocessing import normalize

# ===================== Embedding Generation =====================

def generate_embeddings(X: np.ndarray, model=None, batch_size: int = 128, device: str = "cuda", normalization: bool = True) -> np.ndarray:
    if model is None:
        return X  # passthrough when embeddings are precomputed

    model.eval().to(device)
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            batch_embeddings = model(batch).cpu().numpy()
            embeddings.append(batch_embeddings)
            del batch
            torch.cuda.empty_cache()

    embeddings = np.concatenate(embeddings, axis=0)

    if normalization:
        embeddings = normalize(embeddings, norm='l2', axis=1)

    return embeddings

# ===================== Similarity Calculation =====================

def compute_similarity_scores(emb_enroll, Y_enroll, emb_verify, Y_verify, distance="ed", top_k=40):
    similarity_results = defaultdict(list)
    unique_classes = np.unique(Y_enroll)
    class_indices = [np.where(Y_enroll == cls)[0] for cls in unique_classes]

    if distance == "cd":
        similarity_matrix = -1 * cd(emb_verify, emb_enroll)
    elif distance == "ed":
        similarity_matrix = -1 * ed(emb_verify, emb_enroll)
    else:
        raise ValueError("Unknown distance metric")

    for i in range(similarity_matrix.shape[0]):
        current_class = Y_verify[i]
        scores = similarity_matrix[i]

        for cls in unique_classes:
            indices_cls = class_indices[np.where(unique_classes == cls)[0][0]]
            score = np.mean(sorted(scores[indices_cls], reverse=True)[:top_k])
            label = 1 if current_class == cls else 0
            similarity_results[cls].append([score, label])

    return similarity_results

# ===================== EER Calculation =====================

def compute_eer(result_array):
    result_array = np.array(result_array)
    genuine = result_array[result_array[:, 1] == 1][:, 0]
    impostor = result_array[result_array[:, 1] == 0][:, 0]
    stats = get_eer_stats(genuine, impostor)
    return stats.eer, stats.fmr100


def evaluate_eer_per_class(Y_enroll, similarity_results):
    eer_list = []
    for cls, results in similarity_results.items():
        eer, _ = compute_eer(np.array(results))
        eer_list.append(eer)

    avg_eer = np.mean(eer_list) * 100
    std_eer = np.std(eer_list) * 100
    return avg_eer, std_eer

# ===================== Enrollment Index Split Function =====================

def get_enrollment_verification_indices(Y, S):
    """
    Generate enrollment and verification indices based on first-session enrollment per subject.

    Args:
        Y: numpy array of subject labels
        S: numpy array of session labels

    Returns:
        enroll_idxs: list of indices for enrollment samples
        verify_idxs: list of indices for verification samples
    """
    enroll_idxs, verify_idxs = [], []
    subjects = np.unique(Y)

    for subject in subjects:
        indices = np.where(Y == subject)[0]
        subject_sessions = S[indices]

        if len(np.unique(subject_sessions)) < 2:
            continue

        min_session = np.min(subject_sessions)
        enroll_idxs.extend(indices[subject_sessions == min_session].tolist())
        verify_idxs.extend(indices[subject_sessions != min_session].tolist())

    enroll_idxs = np.random.permutation(enroll_idxs)
    verify_idxs = np.random.permutation(verify_idxs)
    return enroll_idxs, verify_idxs

# ===================== prepare_data_for_evaluation =====================
def prepare_data_for_evaluation(x_e, y_e, s_e, h_e, x_v, y_v, s_v, h_v):
    x_data = np.concatenate([x_e, x_v], axis=0)
    y_data = np.concatenate([y_e, y_v], axis=0)
    s_data = np.concatenate([s_e, s_v], axis=0)
    h_data = np.concatenate([h_e, h_v], axis=0)
    return x_data, y_data, s_data, h_data