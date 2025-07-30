def preprocess_eeg_h5(h5_file_path, normalization='robust', clip_value=5.0, l_freq=1.0, h_freq=50.0, sfreq=500, notch_freqs=[50.0, 60.0], chunk_size=2048):
    import os
    import mne
    import numpy as np
    import torch
    import h5py
    from tqdm import tqdm
    from joblib import Parallel, delayed

    def process_sample(x_i):
        raw = mne.io.RawArray(x_i, info, verbose='ERROR')
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose='ERROR')
        raw.notch_filter(freqs=notch_freqs, verbose='ERROR')
        return raw.get_data()

    def robust_normalize(x):
        median = x.median(dim=2, keepdim=True).values
        q75 = x.quantile(0.75, dim=2, keepdim=True)
        q25 = x.quantile(0.25, dim=2, keepdim=True)
        iqr = q75 - q25
        return (x - median) / (iqr + 1e-6)

    out_path = h5_file_path.replace('.h5', '.h5')
    with h5py.File(h5_file_path, 'r') as src:
        num_samples, num_channels, num_timesteps = src['data'].shape

        with h5py.File(out_path, 'w') as dst:
            for key in src.keys():
                if key != 'data':
                    dst.create_dataset(key, data=src[key])
            dst.create_dataset('data', shape=(num_samples, num_channels, num_timesteps), dtype='float32')

            global info
            info = mne.create_info(ch_names=[f"EEG {i}" for i in range(num_channels)], sfreq=sfreq, ch_types='eeg')

            for start_idx in tqdm(range(0, num_samples, chunk_size), desc=f"Processing {os.path.basename(h5_file_path)}"):
                end_idx = min(start_idx + chunk_size, num_samples)
                data_chunk = src['data'][start_idx:end_idx]  # (chunk, C, T)

                # Parallel filter
                data_filtered = Parallel(n_jobs=75)(delayed(process_sample)(x_i) for x_i in data_chunk)
                data_filtered = np.stack(data_filtered, axis=0)  # (chunk, C, T)
                data_tensor = torch.tensor(data_filtered, dtype=torch.float32)

                if normalization == 'robust':
                    data_tensor = robust_normalize(data_tensor)
                elif normalization == 'clip':
                    mean = data_tensor.mean(dim=2, keepdim=True)
                    std = data_tensor.std(dim=2, keepdim=True)
                    data_tensor = torch.clamp((data_tensor - mean) / (std + 1e-6), -clip_value, clip_value)
                elif normalization == 'zscore':
                    mean = data_tensor.mean(dim=2, keepdim=True)
                    std = data_tensor.std(dim=2, keepdim=True)
                    data_tensor = (data_tensor - mean) / (std + 1e-6)
                elif normalization == 'robust+clip':
                    data_tensor = robust_normalize(data_tensor)
                    data_tensor = torch.clamp(data_tensor, -clip_value, clip_value)
                else:
                    raise ValueError("Normalization must be one of: 'robust', 'clip', 'zscore', 'robust+clip'")

                dst['data'][start_idx:end_idx] = data_tensor.numpy()

    return out_path


# Apply preprocessing to all splits
h5_file_path_valid = '../../Data/valid_raw.h5'
h5_file_path_train = '../../Data/train_raw.h5'
h5_file_path_test  = '../../Data/test_raw.h5'
h5_file_path_neg   = '../../Data/neg_raw.h5'


preprocess_eeg_h5(h5_file_path_valid, normalization='robust+clip')
preprocess_eeg_h5(h5_file_path_train, normalization='robust+clip')
preprocess_eeg_h5(h5_file_path_test, normalization='robust+clip')
preprocess_eeg_h5(h5_file_path_neg, normalization='robust+clip')



import h5py
import numpy as np

# Load data function
def load_data(file_path):
    with h5py.File(file_path, "r") as file:
        data = {
            'X': file['data'][:],
            'Y': file['labels'][:],
            'S': file['sessions'][:],
            'H': file['hardwares'][:]
        }
    return data

# Load and combine datasets
data_test = load_data("../Data/test_raw.h5")
data_neg = load_data("../Data/neg_raw.h5")

# Combine test and negative datasets
for key in data_test:
    data_test[key] = np.concatenate((data_test[key], data_neg[key]), axis=0)

# Shuffle the combined data
indices = np.random.permutation(len(data_test['X']))
for key in data_test:
    data_test[key] = data_test[key][indices]

# Split data based on hardware
def split_and_save_data(data):
    unique_hardware = np.unique(data['H'])
    for hardware in unique_hardware:
        # Filter data for each hardware
        mask = data['H'] == hardware
        filtered_data = {
            'X': data['X'][mask],
            'Y': data['Y'][mask],
            'S': data['S'][mask],
            'H': data['H'][mask]
        }
        
        # Save filtered data to a new HDF5 file
        file_name = f"../Data/test_hardware_{hardware.decode('utf-8') if isinstance(hardware, bytes) else hardware}.h5"
        with h5py.File(file_name, 'w') as f:
            for key, value in filtered_data.items():
                f.create_dataset(key, data=value)
        print(f"Data for hardware {hardware.decode('utf-8') if isinstance(hardware, bytes) else hardware} saved to {file_name}")

split_and_save_data(data_test)


import h5py
import numpy as np

# Load data function
def load_data(file_path):
    with h5py.File(file_path, "r") as file:
        data = {
            'X': file['data'][:],
            'Y': file['labels'][:],
            'S': file['sessions'][:],
            'H': file['hardwares'][:]
        }
    return data

# Load and combine datasets
data_train = load_data("../Data/train_raw.h5")


# Shuffle the combined data
indices = np.random.permutation(len(data_train['X']))
for key in data_train:
    data_train[key] = data_train[key][indices]

# Split data based on hardware
def split_and_save_data(data):
    unique_hardware = np.unique(data['H'])
    for hardware in unique_hardware:
        # Filter data for each hardware
        mask = data['H'] == hardware
        filtered_data = {
            'data': data['X'][mask],
            'labels': data['Y'][mask],
            'sessions': data['S'][mask],
            'H': data['H'][mask]
        }
        
        # Save filtered data to a new HDF5 file
        file_name = f"../Data/train_hardware_{hardware.decode('utf-8') if isinstance(hardware, bytes) else hardware}.h5"
        with h5py.File(file_name, 'w') as f:
            for key, value in filtered_data.items():
                f.create_dataset(key, data=value)
        print(f"Data for hardware {hardware.decode('utf-8') if isinstance(hardware, bytes) else hardware} saved to {file_name}")

split_and_save_data(data_train)



import h5py
import numpy as np
from collections import Counter

# Load the dataset
with h5py.File("../Data/train_raw.h5", "r") as f:
    X_test = f['data'][:]
    Y_test = f['labels'][:]
    S_test = f['sessions'][:]
    H_test = f['hardwares'][:]

# Initialize a new list to store indices to keep
indices_to_keep = []

# Find unique labels
unique_labels = np.unique(Y_test)

# Process each label
for label in unique_labels:
    # Get indices of the current label
    label_indices = np.where(Y_test == label)[0]
    # Extract hardware types for the current label
    hardware_for_label = H_test[label_indices]
    # Count occurrences of each hardware
    hardware_counts = Counter(hardware_for_label)
    # Find the hardware with the majority occurrences
    majority_hardware = max(hardware_counts, key=hardware_counts.get)
    # Keep indices for the majority hardware
    majority_indices = label_indices[hardware_for_label == majority_hardware]
    indices_to_keep.extend(majority_indices)

# Filter the dataset
indices_to_keep = np.array(indices_to_keep)
X_filtered = X_test[indices_to_keep]
Y_filtered = Y_test[indices_to_keep]
S_filtered = S_test[indices_to_keep]
H_filtered = H_test[indices_to_keep]

# Save the filtered dataset
with h5py.File("../Data/train_raw_unconnected.h5", "w") as f_out:
    f_out.create_dataset('data', data=X_filtered)
    f_out.create_dataset('labels', data=Y_filtered)
    f_out.create_dataset('sessions', data=S_filtered)
    f_out.create_dataset('hardwares', data=H_filtered)

print("Filtered dataset saved as train_raw_unconnected.h5.")



import h5py
import numpy as np

# Load data function
def load_data(file_path):
    with h5py.File(file_path, "r") as file:
        data = {
            'X': file['data'][:],
            'Y': file['labels'][:],
            'S': file['sessions'][:],
            'H': file['hardwares'][:]
        }
    return data

# Load and combine datasets
data_test = load_data("../Data/train_raw_unconnected.h5")



# Shuffle the combined data
indices = np.random.permutation(len(data_test['X']))
for key in data_test:
    data_test[key] = data_test[key][indices]

# Split data based on hardware
def split_and_save_data(data):
    unique_hardware = np.unique(data['H'])
    for hardware in unique_hardware:
        # Filter data for each hardware
        mask = data['H'] == hardware
        filtered_data = {
            'data': data['X'][mask],
            'labels': data['Y'][mask],
            'sessions': data['S'][mask],
            'hardwares': data['H'][mask]
        }
        
        # Save filtered data to a new HDF5 file
        file_name = f"../Data/train_unconnected_hardware_{hardware.decode('utf-8') if isinstance(hardware, bytes) else hardware}.h5"
        with h5py.File(file_name, 'w') as f:
            for key, value in filtered_data.items():
                f.create_dataset(key, data=value)
        print(f"Data for hardware {hardware.decode('utf-8') if isinstance(hardware, bytes) else hardware} saved to {file_name}")

split_and_save_data(data_test)
