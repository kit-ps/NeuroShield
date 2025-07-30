import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ===================== Data Loading =====================

x_test_e = np.load('./files/x_test_e.npy')
y_test_e = np.load('./files/y_test_e.npy')
s_test_e = np.load('./files/s_test_e.npy')
h_test_e = np.load('./files/h_test_e.npy')

x_test_v = np.load('./files/x_test_v.npy')
y_test_v = np.load('./files/y_test_v.npy')
s_test_v = np.load('./files/s_test_v.npy')
h_test_v = np.load('./files/h_test_v.npy')

# ===================== Filter BioSemi =====================

def filter_biosemi_data(x_data, y_data, s_data, h_data, excluded_hardware=b'BioSemi'):
    valid_indices = np.where(h_data != excluded_hardware)[0]
    return x_data[valid_indices], y_data[valid_indices], s_data[valid_indices], h_data[valid_indices]

x_test_e, y_test_e, s_test_e, h_test_e = filter_biosemi_data(x_test_e, y_test_e, s_test_e, h_test_e)
x_test_v, y_test_v, s_test_v, h_test_v = filter_biosemi_data(x_test_v, y_test_v, s_test_v, h_test_v)

# ===================== Sampling per subject/session =====================

random.seed(42)

def extract_random_samples_per_subject_session(x_data, y_data, s_data, h_data, selected_subjects, samples_per_session=20, max_session=4):
    x_selected, y_selected, s_selected, h_selected = [], [], [], []

    for subject in selected_subjects:
        subject_indices = np.where(y_data == subject)[0]
        subject_sessions = np.unique(s_data[subject_indices])
        session_counter = 0
        for session in subject_sessions:
            session_counter += 1
            if session_counter > max_session:
                continue
            session_indices = np.where((y_data == subject) & (s_data == session))[0]
            if len(session_indices) > samples_per_session:
                selected_indices = random.sample(list(session_indices), samples_per_session)
            else:
                selected_indices = session_indices
            x_selected.append(x_data[selected_indices])
            y_selected.append(y_data[selected_indices])
            s_selected.append(s_data[selected_indices])
            h_selected.append(h_data[selected_indices])

    x_selected = np.concatenate(x_selected, axis=0)
    y_selected = np.concatenate(y_selected, axis=0)
    s_selected = np.concatenate(s_selected, axis=0)
    h_selected = np.concatenate(h_selected, axis=0)

    return x_selected, y_selected, s_selected, h_selected

# ===================== Subject Selection =====================

num_subjects = 20
unique_subjects = np.unique(y_test_e)
selected_subjects = random.sample(list(unique_subjects), num_subjects)

x_enroll_filtered, y_enroll_filtered, s_enroll_filtered, h_enroll_filtered = extract_random_samples_per_subject_session(
    x_test_e, y_test_e, s_test_e, h_test_e, selected_subjects, samples_per_session=30)

x_verify_filtered, y_verify_filtered, s_verify_filtered, h_verify_filtered = extract_random_samples_per_subject_session(
    x_test_v, y_test_v, s_test_v, h_test_v, selected_subjects, samples_per_session=10)

# ===================== t-SNE =====================

XV1, YV1, S1, H1 = x_enroll_filtered, y_enroll_filtered, s_enroll_filtered, h_enroll_filtered
XV2, YV2, S2, H2 = x_verify_filtered, y_verify_filtered, s_verify_filtered, h_verify_filtered

tsne = TSNE(n_components=2, random_state=42, metric='cosine')
X_combined = np.concatenate([XV1, XV2])
Y_combined = np.concatenate([YV1, YV2])
S_combined = np.concatenate([S1, S2])
X_tsne_combined = tsne.fit_transform(X_combined)

split_idx = XV1.shape[0]
labels = np.unique(Y_combined)

# ===================== Load assigned colors =====================
palette = sns.color_palette('tab20', n_colors=20)
palette = [tuple(color) for color in palette]
labels = np.unique(Y_combined)
assigned_colors2 = {label: palette[i % len(palette)] for i, label in enumerate(labels)}

# ===================== Global Visualization =====================

plt.figure(figsize=(6.3, 4))
marker_list = ['s', 'D', 'P', 'X', '*', 'v', '^', '<', '>']

for i, label in enumerate(labels):
    idx_enroll = (Y_combined == label) & (np.arange(len(Y_combined)) < split_idx)
    points_enroll = X_tsne_combined[idx_enroll]

    idx_verify = (Y_combined == label) & (np.arange(len(Y_combined)) >= split_idx)
    points_verify = X_tsne_combined[idx_verify]
    session_ids = np.unique(S_combined[idx_verify])

    plt.scatter(points_enroll[:, 0], points_enroll[:, 1], label=f"Enroll {label}", 
                color=assigned_colors2[label], marker='o', alpha=1, edgecolors='none')

    for session_idx, session_id in enumerate(session_ids):
        session_points = points_verify[S_combined[idx_verify] == session_id]
        marker = marker_list[session_idx % len(marker_list)]
        plt.scatter(session_points[:, 0], session_points[:, 1],
                    color=assigned_colors2[label], marker=marker, alpha=0.6, edgecolors='k', linewidth=0.5)

plt.title("t-SNE: Enrollment vs Verification")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig('./Results/subjects_20.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== Highlight One Subject =====================

subject_to_highlight = 325
plt.figure(figsize=(6.3, 4))

gray_color = (0.5, 0.5, 0.5)
highlight_palette = sns.color_palette('husl', n_colors=10)
assigned_colors = {label: gray_color for label in labels}
assigned_colors[subject_to_highlight] = assigned_colors2[subject_to_highlight]
random.shuffle(marker_list)

for label in labels:
    idx_enroll = (Y_combined == label) & (np.arange(len(Y_combined)) < split_idx)
    points_enroll = X_tsne_combined[idx_enroll]

    idx_verify = (Y_combined == label) & (np.arange(len(Y_combined)) >= split_idx)
    points_verify = X_tsne_combined[idx_verify]
    session_ids = np.unique(S_combined[idx_verify])

    plt.scatter(points_enroll[:, 0], points_enroll[:, 1],
                label=f"Enroll {label}" if label == subject_to_highlight else None,
                color=assigned_colors[label], marker='o', alpha=1, edgecolors='none')

    if label == subject_to_highlight:
        for session_idx, session_id in enumerate(session_ids):
            session_points = points_verify[S_combined[idx_verify] == session_id]
            session_color = highlight_palette[session_idx % len(highlight_palette)]
            marker = marker_list[session_idx % len(marker_list)]
            plt.scatter(session_points[:, 0], session_points[:, 1],
                        color=session_color, marker=marker, alpha=0.8, edgecolors='k', linewidth=0.5)
    else:
        plt.scatter(points_verify[:, 0], points_verify[:, 1],
                    color=gray_color, marker='x', alpha=0.6, edgecolors='k', linewidth=0.5)

plt.title("t-SNE: Highlight Subject with Sessions")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig('./Results/highlighted_subject_20.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== Magnified View =====================

plt.figure(figsize=(6.3, 4))
highlighted_points = []

for label in labels:
    idx_enroll = (Y_combined == label) & (np.arange(len(Y_combined)) < split_idx)
    points_enroll = X_tsne_combined[idx_enroll]

    idx_verify = (Y_combined == label) & (np.arange(len(Y_combined)) >= split_idx)
    points_verify = X_tsne_combined[idx_verify]
    session_ids = np.unique(S_combined[idx_verify])

    plt.scatter(points_enroll[:, 0], points_enroll[:, 1],
                label=f"Enroll {label}" if label == subject_to_highlight else None,
                color=assigned_colors[label], marker='o', alpha=1, edgecolors='none')

    if label == subject_to_highlight:
        for session_idx, session_id in enumerate(session_ids):
            session_points = points_verify[S_combined[idx_verify] == session_id]
            highlighted_points.append(session_points)
            session_color = highlight_palette[session_idx % len(highlight_palette)]
            marker = marker_list[session_idx % len(marker_list)]
            plt.scatter(session_points[:, 0], session_points[:, 1],
                        color=session_color, marker=marker, alpha=0.8, edgecolors='k', linewidth=0.5)
    else:
        plt.scatter(points_verify[:, 0], points_verify[:, 1],
                    color=gray_color, marker='x', alpha=0.6, edgecolors='k', linewidth=0.5)

highlighted_points = np.vstack(highlighted_points)
plt.xlim(10, 40)
plt.ylim(10, 35)
plt.title("Magnified t-SNE: Highlighted Subject")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig('./Results/highlighted_subject_magnified.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
