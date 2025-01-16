import sys
import os
import subprocess
import gc
import pandas as pd
import mne
import numpy as np
import json
import shutil
import h5py
import os
import numpy as np
import random 
import tempfile
import glob

def check_processed_data_exists(subject_id, session_id):
    """Check if processed data for the given subject, session, and task already exists."""
    # Construct the search pattern for the processed data file
    search_pattern = f'processed_data/subject_{subject_id}_session_{session_id}_task_*'
    
    # Use glob to find any matching files
    matching_files = glob.glob(search_pattern)

    
    # Return True if any matching files are found, otherwise False
    return len(matching_files) > 0


def save_intermediate_results_hdf5(data, subject_id, session_id, item_num, item_name, task, collection_date, hardware_model):
    # Create a directory to store intermediate results if it doesn't exist
    output_dir = 'processed_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scaling factor
    scaling_factor = 1e6

    # Multiply the data by the scaling factor
    scaled_data_segments = np.array(data) * scaling_factor

    # Convert the scaled data to float32
    data_segments_float32 = scaled_data_segments.astype(np.float32)

    # Save to HDF5 file

    if len(data) < 5 :
        print("Skip as data not include enough samples")
    else:    
        save_name = f'subject_{subject_id}_session_{session_id}_task_{task}_date_{collection_date}_hardware_{hardware_model[:7]}_segments.h5'
        save_path = os.path.join(output_dir, save_name)

        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('data_segments', data=data_segments_float32, compression='gzip', compression_opts=9)

def extract_date_from_tsv(tsv_file):
    """Extract the acquisition dates (yyyy-mm-dd) for different tasks from the scans.tsv file."""
    scans_df = pd.read_csv(tsv_file, sep='\t')

    if 'acq_time' in scans_df.columns and len(scans_df) > 0:
        # Create a dictionary to store the date for each task
        date_dict = {}
        for index, row in scans_df.iterrows():
            date_time_str = row['acq_time']
            task_filename = row['filename']
            
            # Extract the task name from the filename (e.g., 'ltpFR2' from '..._task-ltpFR2_eeg.edf')
            if '_task-' in task_filename:
                task_name = task_filename.split('_task-')[1].split('_eeg')[0]
                date_dict[task_name] = date_time_str.split('T')[0]  # Store the date
        return date_dict  # Returns a dictionary of task names and their respective dates
    return None


def extract_hardware_from_json(json_file, mod):
    """Extract hardware model name from the .json file."""
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    if mod == 'a':
        return json_data.get('CapManufacturersModelName', 'UnknownHardware')
    elif mod == 'b':
        return json_data.get('CapManufacturer', 'UnknownHardware')


def download_all_files(subject_id, session_id, bucket_url, download_dir):
    """Download all files from the EEG folder in the S3 bucket for the given subject and session."""
    subject_str = f"sub-LTP{subject_id:03d}"
    session_str = f"ses-{session_id:00d}"
    
    # Create directory if it doesn't exist
    subject_session_dir = os.path.join(download_dir, subject_str, session_str)
    os.makedirs(subject_session_dir, exist_ok=True)

    # Define the S3 folder URL
    s3_url = f"{bucket_url}/{subject_str}/{session_str}/"
    
    try:
        # Use the aws s3 sync command to download all files from the directory
        subprocess.run(["aws", "s3", "sync", "--no-sign-request", "--quiet", s3_url, subject_session_dir], check=True)
        print(f"Downloaded all files for {subject_str}, {session_str}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading files for {subject_str}, {session_str}: {e}")
        return False  # Indicate failure if download fails

    return True  # Return True if the download succeeds


def find_all_task_files(subject_id, session_id, download_dir):
    """Find all EEG, event, and CapTrak files for a given task."""
    task_names = ['ltpFR', 'ltpFR2', 'VFFR']
    subject_str = f"sub-LTP{subject_id:03d}"
    session_str = f"ses-{session_id:00d}"

    task_files = []

    for task in task_names:
        # Construct potential file paths
        if task == 'VFFR':
            edf_path = f'{download_dir}/{subject_str}/{session_str}/eeg/sub-LTP{subject_id:03d}_ses-{session_id:00d}_task-{task}_eeg.bdf'
        else:
            edf_path = f'{download_dir}/{subject_str}/{session_str}/eeg/sub-LTP{subject_id:03d}_ses-{session_id:00d}_task-{task}_eeg.edf'
        event_path = f'{download_dir}/{subject_str}/{session_str}/eeg/sub-LTP{subject_id:03d}_ses-{session_id:00d}_task-{task}_events.tsv'
        captrak_path = f'{download_dir}/{subject_str}/{session_str}/eeg/sub-LTP{subject_id:03d}_ses-{session_id:00d}_space-CapTrak_electrodes.tsv'

        # Check if the files exist
        if os.path.exists(edf_path) and os.path.exists(event_path):
            print(f"Found task {task} for subject {subject_id}, session {session_id}")
            task_files.append((edf_path, event_path, captrak_path, task))

    return task_files


import os

def find_all_task_files(subject_id, session_id, download_dir):
    """Find all EEG, event, and CapTrak files for a given task, automatically detecting .bdf or .edf."""
    task_names = ['ltpFR', 'ltpFR2', 'VFFR']
    subject_str = f"sub-LTP{subject_id:03d}"
    session_str = f"ses-{session_id:00d}"

    task_files = []

    for task in task_names:
        # Construct base path for the EEG file
        base_eeg_path = os.path.join(download_dir, subject_str, session_str, 'eeg',
                                     f"{subject_str}_ses-{session_id:00d}_task-{task}_eeg")
        
        # Check if the file exists in either .bdf or .edf format
        eeg_path = None
        if os.path.exists(base_eeg_path + '.bdf'):
            eeg_path = base_eeg_path + '.bdf'
        elif os.path.exists(base_eeg_path + '.edf'):
            eeg_path = base_eeg_path + '.edf'
        else:
            print(f"Warning: No .bdf or .edf EEG file found for {task}, subject {subject_id}, session {session_id}. Skipping.")
            continue
        
        # Construct paths for the event and CapTrak files
        event_path = os.path.join(download_dir, subject_str, session_str, 'eeg',
                                  f"{subject_str}_ses-{session_id:00d}_task-{task}_events.tsv")
        captrak_path = os.path.join(download_dir, subject_str, session_str, 'eeg',
                                    f"{subject_str}_ses-{session_id:00d}_space-CapTrak_electrodes.tsv")

        # Check if event file exists (CapTrak is optional)
        if os.path.exists(event_path):
            print(f"Found task {task} for subject {subject_id}, session {session_id}")
            task_files.append((eeg_path, event_path, captrak_path if os.path.exists(captrak_path) else None, task))
        else:
            print(f"Warning: Event file missing for {task}, subject {subject_id}, session {session_id}. Skipping.")

    return task_files


def process_subject_data(subject_id, session_id, captrak_path, edf_path, event_path, task, hardware_model):

    eeg_clist = ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2',
                 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
                 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
                 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4',
                 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9',
                 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'Oz', 'O2',
                 'O9', 'Iz', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2']


    print(f"Processing task {task} for subject {subject_id}, session {session_id}")


    if hardware_model[0:7] == 'Geodisi':
        mapping = {'E128': 'Fp1', 'E17': 'Fpz', 'E125': 'Fp2', 'E44': 'AF9', 'E39': 'AF7', 'E33': 'AF5', 'E27': 'AF3', 'E22': 'AF1', 'E14': 'AFz', 'E1': 'AF2', 'E2': 'AF4', 'E121': 'AF6', 'E120': 'AF8', 'E114': 'AF10', 'E49': 'F9', 'E45': 'F7', 'E40': 'F5', 'E34': 'F3', 'E28': 'F1', 'E10': 'Fz', 'E122': 'F2', 'E116': 'F4', 'E115': 'F6', 'E109': 'F8', 'E108': 'F10', 'E56': 'FT9', 'E46': 'FT7', 'E41': 'FC5', 'E35': 'FC3', 'E36': 'FC1', 'E107': 'FCz', 'E117': 'FC2', 'E110': 'FC4', 'E102': 'FC6', 'E101': 'FT8', 'E100': 'FT10', 'E63': 'T9', 'E57': 'T7', 'E50': 'C5', 'E47': 'C3', 'E48': 'C1', 'E99': 'C2', 'E103': 'C4', 'E97': 'C6', 'E96': 'T8', 'E95': 'T10', 'E69': 'TP9', 'E64': 'TP7', 'E58': 'CP5', 'E59': 'CP3', 'E60': 'CP1', 'E73': 'CPz', 'E86': 'CP2', 'E92': 'CP4', 'E98': 'CP6', 'E90': 'TP8', 'E89': 'TP10', 'E74': 'P9', 'E70': 'P7', 'E65': 'P5', 'E71': 'P3', 'E66': 'P1', 'E76': 'Pz', 'E85': 'P2', 'E91': 'P4', 'E83': 'P6', 'E82': 'P8', 'E84': 'P10', 'E75': 'PO9', 'E72': 'PO7', 'E51': 'PO5', 'E77': 'PO3', 'E67': 'PO1', 'E78': 'POz', 'E68': 'PO2', 'E93': 'PO4', 'E79': 'PO6', 'E87': 'PO8', 'E104': 'PO10', 'E52': 'O1', 'E61': 'Oz', 'E62': 'O2', 'E53': 'O9', 'E42': 'Iz', 'E94': 'O10', 'E43': 'T3', 'E37': 'T5', 'E111': 'T4', 'E105': 'T6', 'E29': 'M1', 'E123': 'M2', 'E23': 'A1', 'E118': 'A2'}
    elif hardware_model[0:7] == 'HydroCe':
        mapping = {'E17': 'Fp1', 'E14': 'Fpz', 'E1': 'Fp2', 'E48': 'AF9', 'E128': 'AF7', 'E32': 'AF5', 'E26': 'AF3', 'E21': 'AF1', 'E15': 'AFz', 'E9': 'AF2', 'E2': 'AF4', 'E125': 'AF6', 'E121': 'AF8', 'E119': 'AF10', 'E43': 'F9', 'E38': 'F7', 'E33': 'F5', 'E27': 'F3', 'E23': 'F1', 'E10': 'Fz', 'E123': 'F2', 'E122': 'F4', 'E120': 'F6', 'E114': 'F8', 'E113': 'F10', 'E49': 'FT9', 'E44': 'FT7', 'E39': 'FC5', 'E40': 'FC3', 'E35': 'FC1', 'E106': 'FCz', 'E110': 'FC2', 'E115': 'FC4', 'E108': 'FC6', 'E107': 'FT8', 'E99': 'FT10', 'E56': 'T9', 'E63': 'T7', 'E45': 'C5', 'E46': 'C3', 'E47': 'C1', 'E98': 'C2', 'E101': 'C4', 'E100': 'C6', 'E95': 'T8', 'E94': 'T10', 'E68': 'TP9', 'E57': 'TP7', 'E64': 'CP5', 'E58': 'CP3', 'E59': 'CP1', 'E72': 'CPz', 'E84': 'CP2', 'E96': 'CP4', 'E89': 'CP6', 'E88': 'TP8', 'E81': 'TP10', 'E73': 'P9', 'E69': 'P7', 'E65': 'P5', 'E70': 'P3', 'E66': 'P1', 'E75': 'Pz', 'E83': 'P2', 'E90': 'P4', 'E82': 'P6', 'E74': 'P8', 'E97': 'P10', 'E50': 'PO9', 'E51': 'PO7', 'E71': 'PO5', 'E76': 'PO3', 'E67': 'PO1', 'E77': 'POz', 'E91': 'PO2', 'E85': 'PO4', 'E102': 'PO6', 'E92': 'PO8', 'E109': 'PO10', 'E60': 'O1', 'E52': 'Oz', 'E78': 'O2', 'E61': 'O9', 'E53': 'Iz', 'E103': 'O10', 'E34': 'T3', 'E41': 'T5', 'E116': 'T4', 'E93': 'T6', 'E42': 'M1', 'E117': 'M2', 'E22': 'A1', 'E3': 'A2'}
    elif hardware_model[0:7] == 'BioSemi':
        mapping = {'C29': 'Fp1', 'C17': 'Fpz', 'C16': 'Fp2', 'C30': 'AF9', 'D7': 'AF7', 'C31': 'AF5', 'C28': 'AF3', 'C18': 'AF1', 'C15': 'AFz', 'C8': 'AF2', 'C9': 'AF4', 'C7': 'AF6', 'B27': 'AF8', 'C6': 'AF10', 'D8': 'F9', 'D23': 'F7', 'D6': 'F5', 'C32': 'F3', 'C27': 'F1', 'C19': 'Fz', 'C10': 'F2', 'C5': 'F4', 'B28': 'F6', 'B26': 'F8', 'B14': 'F10', 'D24': 'FT9', 'D9': 'FT7', 'D22': 'FC5', 'D10': 'FC3', 'D3': 'FC1', 'C22': 'FCz', 'C3': 'FC2', 'B29': 'FC4', 'B25': 'FC6', 'B10': 'FT8', 'B9': 'FT10', 'D32': 'T9', 'D31': 'T7', 'D25': 'C5', 'D26': 'C3', 'D17': 'C1', 'B18': 'C2', 'B16': 'C4', 'B15': 'C6', 'B11': 'T8', 'B8': 'T10', 'A12': 'TP9', 'A11': 'TP7', 'D30': 'CP5', 'D29': 'CP3', 'A6': 'CP1', 'A4': 'CPz', 'B3': 'CP2', 'B13': 'CP4', 'B12': 'CP6', 'B7': 'TP8', 'A26': 'TP10', 'A13': 'P9', 'A10': 'P7', 'A9': 'P5', 'A8': 'P3', 'A17': 'P1', 'A20': 'Pz', 'A30': 'P2', 'B6': 'P4', 'A28': 'P6', 'A27': 'P8', 'A25': 'P10', 'A14': 'PO9', 'A15': 'PO7', 'A16': 'PO5', 'A23': 'PO3', 'A22': 'PO1', 'A21': 'POz', 'A29': 'PO2', 'A24': 'PO4', 'B5': 'PO6', 'A31': 'PO8', 'B4': 'PO10', 'A18': 'O1', 'A19': 'Oz', 'A32': 'O2', 'A7': 'O9', 'A5': 'Iz', 'B17': 'O10', 'D21': 'T3', 'D27': 'T5', 'B24': 'T4', 'B23': 'T6', 'D20': 'M1', 'B30': 'M2', 'D5': 'A1', 'B22': 'A2'}
    else:
        print("we could not detect device layout")
    
    # Load the EEG data (from the .edf file)
    if edf_path.endswith('.bdf'):
        raw = mne.io.read_raw_bdf(edf_path, preload=True, verbose=False)
    elif edf_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {edf_path}")

    # Function to remove prefixes like 'EEG ' or 'EOG '
    def remove_prefix(channel_name):
        if channel_name.startswith('EEG ') or channel_name.startswith('EOG '):
            return channel_name.split(' ')[1]  # Remove 'EEG ' or 'EOG ' prefix
        return channel_name

    # Apply the function to all channel names to remove 'EEG ' or 'EOG ' prefix
    cleaned_raw_channel_names = [remove_prefix(ch_name) for ch_name in raw.info['ch_names']]

    # Create a dictionary mapping original channel names to cleaned names
    mapping2 = {old_name: new_name for old_name, new_name in zip(raw.info['ch_names'], cleaned_raw_channel_names)}


    #print("mapping2", mapping2)
    
    # Rename the channels in the raw data
    raw.rename_channels(mapping2)
    
    # Apply the CapTrak to 10-20 system mapping
    #print("mapping", mapping)
    #raw.rename_channels(mapping)

    # Step 2: Intermediate step: rename all raw channels to unique temporary names to avoid conflicts
    temp_mapping = {ch_name: f"Temp{i+1}" for i, ch_name in enumerate(raw.info['ch_names'])}
    raw.rename_channels(temp_mapping)

    #print("temp_mapping mapping:", temp_mapping, len(temp_mapping))

    # Step 3: Apply the CapTrak to 10-20 system mapping using the temporary names
    final_mapping = {temp_mapping[original]: mapped for original, mapped in mapping.items() if original in temp_mapping}

    #print("Final mapping:", final_mapping, len(final_mapping))

    # Step 4: Rename channels using the final mapping
    raw.rename_channels(final_mapping)
    

    # Reorder the channels as per the 10-20 system
    #standard_1020_list = list(xyz_coords_1020.keys())  # Already excludes 'Cz'
    raw_63 = raw.pick_channels(eeg_clist)

    # Read the events file and process the segments
    events_df = pd.read_csv(event_path, delimiter='\t')

    # Get event sample, item names, and item numbers
    item_num = np.array(events_df[events_df['trial_type'] == 'WORD']['item_num']).tolist()
    item_name = np.array(events_df[events_df['trial_type'] == 'WORD']['item_name']).tolist()
    events_sample = np.array(events_df[events_df['trial_type'] == 'WORD']['sample'])

    if len(events_sample) < 1:
        print(subject_id, session_id, "Invalid events_df", len(events_sample))
        return None, None, None, None, None

    
    raw_63 = raw_63.reorder_channels(eeg_clist)
    raw_63, _ = mne.set_eeg_reference(raw_63, verbose=False)
    raw_63 = raw_63.filter(l_freq=0.1, h_freq=60, verbose=False)

    
    #print("data shape 0 ", raw_63.get_data().shape)
    # Process the segmented data
    data_segments, subject_ids, session_ids = make_raw_array(raw_63, subject_id, session_id, events_sample)

    return data_segments, subject_ids, session_ids, item_num, item_name


def make_raw_array(raw, subject_id, session_id, induce):
    eeg_data = raw.get_data()
    print("data shape", eeg_data.shape)

    segments = []
    data_length = eeg_data.shape[1]
    for start_index in induce:
        if start_index < 1:
            continue
        if start_index + 500 < data_length:  # Ensure the segment does not exceed the data length
            segment = eeg_data[:, start_index:start_index + 500]
            segments.append(segment)
        else:
            print(f"Segment starting at {start_index} goes out of bounds.")

    
    if len(segments) > 100:
        segments = random.sample(segments, 100)
    subject_ids = [subject_id] * len(segments)
    session_ids = [session_id] * len(segments)

    print("segments:", subject_id, session_id, ":", len(segments))
    return segments, subject_ids, session_ids

def save_intermediate_results(data, subject_id, session_id, item_num, item_name, task, collection_date, hardware_model):
    # Create a directory to store intermediate results if it doesn't exist
    output_dir = 'processed_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scaling factor
    scaling_factor = 1e6

    # Multiply the data by the scaling factor
    scaled_data_segments = np.array(data) * scaling_factor

    # Convert the scaled data to float32
    data_segments_float32 = scaled_data_segments.astype(np.float32)

    # Save segmented data to disk with task name, collection date, and hardware model
    save_name = f'subject_{subject_id}_session_{session_id}_task_{task}_date_{collection_date}_hardware_{hardware_model[:7]}_segments.npy'
    np.save(os.path.join(output_dir, save_name), data_segments_float32)

    # Save item numbers and names
    #np.save(os.path.join(output_dir, f'subject_{subject_id}_session_{session_id}_task_{task}_item_num.npy'), item_num)
    #np.save(os.path.join(output_dir, f'subject_{subject_id}_session_{session_id}_task_{task}_item_name.npy'), item_name)


if __name__ == '__main__':
    # Check if we are in a Jupyter notebook or standard Python execution
    if 'ipykernel' in sys.modules:
        # Default values for Jupyter notebook execution
        print("Running in a Jupyter notebook")
        subject_id = 63
        session_id = 1
    else:
        # Command-line execution, handle arguments
        if len(sys.argv) < 3:
            print("Error: Please provide subject_id and session_id as arguments.")
            sys.exit(1)
        
        try:
            # Read subject_id and session_id from command-line arguments
            subject_id = int(sys.argv[1])
            session_id = int(sys.argv[2])
        except (ValueError, IndexError):
            print("Error: Invalid subject_id or session_id provided.")
            sys.exit(1)

    # Set the bucket URL and download directory
    bucket_url = "s3://openneuro.org/ds004395"
    #download_dir = "ds004395-download"
    download_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
    download_dir = download_dir + "/matin"

    print("URL", download_dir)


    if check_processed_data_exists(subject_id, session_id):
        print(f"Processed data for subject {subject_id}, session {session_id}. (already exist)")
    else:
        print("download is started", subject_id, session_id)
        # Download necessary files for the given subject and session
        download_successful = download_all_files(subject_id, session_id, bucket_url, download_dir)
        
        if not download_successful:
            print(f"Failed to download required files for subject {subject_id}, session {session_id}")
            sys.exit(1)  # Exit if download fails
    
        # Extract date from the scans.tsv file
        scans_tsv_path = f'{download_dir}/sub-LTP{subject_id:03d}/ses-{session_id:00d}/sub-LTP{subject_id:03d}_ses-{session_id:00d}_scans.tsv'
        date_dict = extract_date_from_tsv(scans_tsv_path)
    
        # Find the appropriate task files (ltpFR, ltpFR2, VFFR)
        task_files = find_all_task_files(subject_id, session_id, download_dir)
    
        if len(task_files) == 0:
            print(f"No valid task files found for subject {subject_id}, session {session_id}")
            sys.exit(1)  # Exit if no task files are found
    
        # Process each task's data
        for edf_path, event_path, captrak_path, task in task_files:
            # Extract hardware model from corresponding json file
    
            if edf_path.endswith('.bdf'):
                json_file_path = edf_path.replace('_eeg.bdf', '_eeg.json')
                hardware_model = extract_hardware_from_json(json_file_path, 'b')  # Assuming 'b' for bdf
            elif edf_path.endswith('.edf'):
                json_file_path = edf_path.replace('_eeg.edf', '_eeg.json')
                hardware_model = extract_hardware_from_json(json_file_path, 'a')  # Assuming 'a' for edf
            else:
                raise ValueError(f"Unsupported file format: {edf_path}")
    
            collection_date = date_dict.get(task, None)  # Match task name with the dictionary
    
            if collection_date is None:
                print(f"Warning: No date found for task {task} in subject {subject_id}, session {session_id}")
                continue  # Skip this task if no matching date is found
    
    
            
            print("Hardware:", hardware_model)
            data, sub_id, se_id, item_num, item_name = process_subject_data(subject_id, session_id, captrak_path, edf_path, event_path, task, hardware_model)
    
            if data is not None:
                # Save intermediate results to disk
                save_intermediate_results_hdf5(data, subject_id, session_id, item_num, item_name, task, collection_date, hardware_model)
    
    
            # Delete the folder after processing
        subject_session_dir = os.path.join(download_dir, f"sub-LTP{subject_id:03d}", f"ses-{session_id:00d}")
        if os.path.exists(subject_session_dir):
            print(f"Deleting folder: {subject_session_dir}")
            shutil.rmtree(subject_session_dir)
            print(f"Deleted {subject_session_dir} successfully.")


    # Release memory explicitly
    gc.collect()

