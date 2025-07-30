import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== Helper Functions =====================

def compute_intra_inter_distances(distance_matrix, labels):
    unique_subjects = np.unique(labels)
    intra_distances = []
    inter_distances = []

    for subject in unique_subjects:
        subject_indices = np.where(labels == subject)[0]
        other_indices = np.where(labels != subject)[0]

        if len(subject_indices) > 1:
            intra = distance_matrix[np.ix_(subject_indices, subject_indices)]
            intra_distances.extend(intra[np.triu_indices(len(subject_indices), k=1)])

        inter = distance_matrix[np.ix_(subject_indices, other_indices)]
        inter_distances.extend(inter.flatten())

    avg_intra = np.mean(intra_distances) if intra_distances else 0
    avg_inter = np.mean(inter_distances) if inter_distances else 0

    return avg_intra, avg_inter, avg_inter / avg_intra if avg_intra > 0 else np.inf


def compute_closeness(distance_matrix, labels, k=5):
    closeness = []
    for i, subject in enumerate(labels):
        same_subject_indices = np.where(labels == subject)[0]
        if len(same_subject_indices) > 1:
            distances_to_same = distance_matrix[i, same_subject_indices]
            distances_to_same = distances_to_same[distances_to_same > 0]
            if len(distances_to_same) >= k:
                closeness.append(np.mean(np.sort(distances_to_same)[:k]))
            else:
                closeness.append(np.mean(distances_to_same))
        else:
            closeness.append(np.nan)
    return np.array(closeness)

# ===================== Data Loading =====================

x_test_e = np.load('./files/x_test_e.npy')
y_test_e = np.load('./files/y_test_e.npy')
s_test_e = np.load('./files/s_test_e.npy')
h_test_e = np.load('./files/h_test_e.npy')

x_test_v = np.load('./files/x_test_v.npy')
y_test_v = np.load('./files/y_test_v.npy')
s_test_v = np.load('./files/s_test_v.npy')
h_test_v = np.load('./files/h_test_v.npy')

x_test = np.concatenate((x_test_e, x_test_v), axis=0)
y_test = np.concatenate((y_test_e, y_test_v), axis=0)
s_test = np.concatenate((s_test_e, s_test_v), axis=0)
h_test = np.concatenate((h_test_e, h_test_v), axis=0)

print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"s_test shape: {s_test.shape}")
print(f"h_test shape: {h_test.shape}")

# ===================== Sampling =====================

unique_sessions = np.unique(s_test)
unique_subjects = np.unique(y_test)
selected_indices = []

for session in unique_sessions:
    for subject in unique_subjects:
        session_subject_indices = np.where((s_test == session) & (y_test == subject))[0]
        if len(session_subject_indices) > 30:
            selected_indices.extend(np.random.choice(session_subject_indices, 30, replace=False))
        elif len(session_subject_indices) > 0:
            selected_indices.extend(session_subject_indices)

x_test_selected = x_test[selected_indices]
y_test_selected = y_test[selected_indices]
s_test_selected = s_test[selected_indices]
h_test_selected = h_test[selected_indices]

print(f"x_test_selected shape: {x_test_selected.shape}")
print(f"Number of unique subjects: {len(np.unique(y_test_selected))}")
print(f"Number of unique sessions: {len(np.unique(s_test_selected))}")

# ===================== t-SNE =====================

tsne = TSNE(n_components=2, random_state=42, metric="cosine")
x_tsne = tsne.fit_transform(x_test_selected)
print(f"t-SNE output shape: {x_tsne.shape}")

# ===================== Distance Matrix =====================

dist = cosine_distances(x_test_selected)
print(f"Cosine distance matrix shape: {euclidean_dist.shape}")

# ===================== Closeness & Distinguishability =====================

closeness = compute_closeness(dist, y_test_selected, k=5)
avg_intra_dist, avg_inter_dist, distinguishability_ratio = compute_intra_inter_distances(dist, y_test_selected)

print(f"Avg Intra-Subject Cosine Distance: {avg_intra_dist:.3f}")
print(f"Avg Inter-Subject Cosine Distance: {avg_inter_dist:.3f}")
print(f"Distinguishability Ratio (Inter/Intra): {distinguishability_ratio:.3f}")


# ===================== Closeness Visualization =====================

df_viz = pd.DataFrame({
    "y": x_tsne[:, 0],
    "x": x_tsne[:, 1],
    "Closeness": closeness,
    "Subject": y_test_selected
})

plt.figure(figsize=(6.3, 4))
scatter = plt.scatter(
    df_viz["x"], df_viz["y"], c=df_viz["Closeness"],
    cmap="coolwarm",  # Colorblind-friendly continuous colormap
    alpha=0.6, s=10, vmin=0, vmax=1.5  # Adjust vmax depending on your closeness scale
)
plt.colorbar(scatter, label="Closeness")
plt.title(f"t-SNE of Embeddings\nIntra-Dist: {avg_intra_dist:.3f}, Inter-Dist: {avg_inter_dist:.3f}")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.savefig("./Results/embedding_space_closeness.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== Visualization Device =====================

h_test_selected = np.array([device.decode('utf-8') if isinstance(device, bytes) else device for device in h_test_selected])
unique_devices = np.unique(h_test_selected)
device_colors = {device: color for device, color in zip(unique_devices, sns.color_palette("colorblind", len(unique_devices)))}

plt.figure(figsize=(6, 4))
for device in unique_devices:
    indices = np.where(h_test_selected == device)[0]
    plt.scatter(x_tsne[indices, 1], x_tsne[indices, 0], 
                label=f"{device}", color=device_colors[device], alpha=0.5, s=10)

plt.title("t-SNE Visualization of Embeddings Colored by Device Type")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, markerscale=4)

plt.savefig("./Results/embedding_space_device.png", dpi=300, bbox_inches='tight')
plt.show()
