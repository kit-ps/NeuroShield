import os
import re
from collections import defaultdict
import random

# Function to extract metadata from filename
def extract_info_from_filename(filename):
    pattern = r"subject_(\d+)_session_(\d+)_task_(\w+)_date_(\d{4}-\d{2}-\d{2})_hardware_(\w+)_segments.h5"
    match = re.match(pattern, filename)
    if match:
        subject = int(match.group(1))
        session = int(match.group(2))
        task = match.group(3)
        date = match.group(4)
        hardware = match.group(5)
        return subject, session, task, date, hardware
    else:
        print("NOT Found: Pattern")
    return None, None, None, None, None

# List files and extract metadata
def list_files_and_extract_metadata(directory):
    filenames = os.listdir(directory)
    data_info = [extract_info_from_filename(f) for f in filenames]
    return filenames, data_info

# Group files by subject and session
def group_files_by_subject(data_info):
    subjects_sessions = defaultdict(list)
    for subject, session, task, date, hardware in data_info:
        if subject is not None:
            subjects_sessions[subject].append((session, task, date, hardware))
    return subjects_sessions

# Main workflow
directory = './processed_data/'

# Step 1: List files and extract metadata
filenames, data_info = list_files_and_extract_metadata(directory)

# Step 2: Group the files by subject and session
subjects_sessions = group_files_by_subject(data_info)

# Step 3: Split into multi-session and single-session subjects
multi_session_subjects = {s: sessions for s, sessions in subjects_sessions.items() if len(sessions) > 1}
single_session_subjects = {s: sessions for s, sessions in subjects_sessions.items() if len(sessions) == 1}

# Step 4: Select 50 test subjects and 10 validation subjects from multi-session subjects
random.seed(42)
test_subjects = random.sample(list(multi_session_subjects.keys()), 100)
remaining_subjects = [s for s in multi_session_subjects if s not in test_subjects]
validation_subjects = random.sample(remaining_subjects, 15)
train_subjects = [s for s in remaining_subjects if s not in validation_subjects]

# Add single-session subjects to the training set
#train_subjects.extend(single_session_subjects.keys())
print("single session number", len(single_session_subjects.keys()))

# Function to group files by subject lists
def group_files_by_split(subject_list, data_info, filenames):
    files = []
    for subject in subject_list:
        files.extend([filenames[i] for i, (s, _, _, _, _) in enumerate(data_info) if s == subject])
    return files

# Step 5: Group the files for each split (train, validation, and test)
train_files = group_files_by_split(train_subjects, data_info, filenames)
valid_files = group_files_by_split(validation_subjects, data_info, filenames)
Negative_files = group_files_by_split(test_subjects[:50], data_info, filenames)
test_files = group_files_by_split(test_subjects[50:], data_info, filenames)

print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(valid_files)}")
print(f"Negative files: {len(Negative_files)}")
print(f"Test files: {len(test_files)}")


import h5py
import numpy as np
import os

def create_label(subject, session, task, date, hardware):
    # You can use different strategies here to create labels.
    return subject  # This is currently returning only the subject ID

def concatenate_h5_files(file_list, directory, output_file):
    """
    Concatenates multiple h5 files from a directory into a single h5 file.
    Additionally saves session, task, date, and hardware information.
    
    file_list: List of input h5 filenames (without full path).
    directory: Directory where the h5 files are located.
    output_file: Path to the output h5 file.
    """
    all_labels = []
    all_sessions = []
    all_tasks = []
    all_dates = []
    all_hardwares = []

    # Initialize output file and datasets
    with h5py.File(output_file, 'w') as f_out:
        data_dset = None  # To hold the dataset for the concatenated data

        for file in file_list:
            # Create the full path for the file
            full_path = os.path.join(directory, file)

            # Check if the file exists
            if not os.path.exists(full_path):
                print(f"Warning: File {full_path} does not exist. Skipping.")
                continue

            with h5py.File(full_path, 'r') as f:
                # Assuming 'data_segments' or 'data' as keys for EEG data
                if 'data_segments' in f:
                    data = f['data_segments'][:]
                elif 'data' in f:
                    data = f['data'][:]
                else:
                    print(f"Warning: No valid data found in {file}. Skipping.")
                    continue

                # Check if the data has the expected number of dimensions
                if data.ndim != 3:
                    print(f"Skipping file {file} due to unexpected data shape: {data.shape}")
                    continue

                # Initialize the dataset in the output file with the correct shape only once
                if data_dset is None:
                    all_data_shape = (0, data.shape[1], data.shape[2])  # Use the shape from the first file
                    data_dset = f_out.create_dataset(
                        'data', shape=all_data_shape, maxshape=(None, data.shape[1], data.shape[2]), chunks=True
                    )
                
                # Resize the dataset to accommodate new data
                current_size = data_dset.shape[0]
                new_size = current_size + data.shape[0]
                data_dset.resize(new_size, axis=0)
                data_dset[current_size:new_size] = data  # Append the new data

                # Extract metadata from the filename and generate labels
                subject, session, task, date, hardware = extract_info_from_filename(file)
                label = create_label(subject, session, task, date, hardware)

                # Extend metadata lists
                all_labels.extend([label] * len(data))  # Extend label list
                all_sessions.extend([session] * len(data))  # Save session for each data point
                all_tasks.extend([task] * len(data))  # Save task for each data point
                all_dates.extend([date] * len(data))  # Save date for each data point
                all_hardwares.extend([hardware] * len(data))  # Save hardware for each data point

        # Now write all the metadata once
        if len(all_labels) > 0:
            f_out.create_dataset('labels', data=np.array(all_labels))
            f_out.create_dataset('sessions', data=np.array(all_sessions))
            f_out.create_dataset('tasks', data=np.array(all_tasks, dtype="S"))  # Save task as string
            f_out.create_dataset('dates', data=np.array(all_dates, dtype="S"))  # Save date as string
            f_out.create_dataset('hardwares', data=np.array(all_hardwares, dtype="S"))  # Save hardware as string

        print(f"Successfully created {output_file} with shape: {data_dset.shape}, labels: {len(all_labels)}")



# Directory where the files are stored
directory = './processed_data/'

concatenate_h5_files(valid_files, directory, '../Data/validation.h5')
concatenate_h5_files(Negative_files, directory, '../Data/Negative.h5')
concatenate_h5_files(train_files, directory, '../Data/train.h5')
concatenate_h5_files(test_files, directory, '../Data/test.h5')

