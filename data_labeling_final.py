import os
import mne
import numpy as np
import pandas as pd

# Constants
EPOCH_LENGTH = 30
WINDOW_SIZE = 2
ALERT_RT_PERCENTILE = 5
ALERT_RT_MULTIPLIER = 1.5
FATIGUE_RT_MULTIPLIER = 2.0

# Directory containing all .set files
directory = "preprocessed_data"

# DataFrame to hold results from all datasets
all_data = pd.DataFrame(columns=['Epoch', 'Local RT', 'Global RT', 'Label'])


def calculate_global_rt(local_rt):
    # Calculate global RT for each trial: average of local RTs within the window
    return [np.mean(local_rt[max(0, i - WINDOW_SIZE): min(len(local_rt), i + WINDOW_SIZE)]) for i in
            range(len(local_rt))]


def calculate_labels(local_rt, global_rt, alert_rt):
    # Label each trial based on local RT and global RT compared to alert RT
    return ['alert' if lr < alert_rt * ALERT_RT_MULTIPLIER and gr < alert_rt * ALERT_RT_MULTIPLIER else
            'fatigued' if lr > alert_rt * FATIGUE_RT_MULTIPLIER and gr > alert_rt * FATIGUE_RT_MULTIPLIER else
            'unknown' for lr, gr in zip(local_rt, global_rt)]


def process_set_file(file_path):
    # Load the EEG data
    try:
        eeg_data = mne.io.read_raw_eeglab(file_path, preload=True)
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None

    # Extract events
    events, event_id = mne.events_from_annotations(eeg_data)

    # Number of samples per epoch
    samples_per_epoch = int(EPOCH_LENGTH * eeg_data.info['sfreq'])

    # Number of epochs in the dataset
    n_epochs = eeg_data.n_times // samples_per_epoch

    local_rt_list = []
    for i_epoch in range(n_epochs):
        # Indices of events that occur within this epoch
        epoch_event_indices = np.where((events[:, 0] >= i_epoch * samples_per_epoch) &
                                       (events[:, 0] < (i_epoch + 1) * samples_per_epoch))[0]

        # Ensure the indices do not go beyond the last event
        valid_indices = epoch_event_indices[epoch_event_indices < len(events) - 1]

        # Calculate local RT for this epoch
        local_rt = np.sum(events[valid_indices + 1, 0] - events[valid_indices, 0]) / eeg_data.info['sfreq']
        local_rt_list.append(local_rt)

    # Calculate alert RT: the ALERT_RT_PERCENTILE percentile of all local RTs
    alert_rt = np.percentile(local_rt_list, ALERT_RT_PERCENTILE)

    # Calculate global RT for each trial
    global_rt_list = calculate_global_rt(local_rt_list)

    # Label each trial
    labels = calculate_labels(local_rt_list, global_rt_list, alert_rt)

    # Create a DataFrame for easier viewing
    df = pd.DataFrame({
        'Epoch': np.arange(1, n_epochs + 1),
        'Local RT': local_rt_list,
        'Global RT': global_rt_list,
        'Label': labels,
    })

    return df


# Iterate over all .set files in the main directory and its subdirectories
for subdir, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".set"):
            set_file_path = os.path.join(subdir, file)
            df = process_set_file(set_file_path)

            if df is not None:
                print(df)

                # Calculate proportions of each label
                label_counts = df['Label'].value_counts(normalize=True)
                print(label_counts)

                # Append the results to the all_data DataFrame
                all_data = pd.concat([all_data, df], ignore_index=True)

# Calculate proportions of each label for all data
label_counts_all = all_data['Label'].value_counts(normalize=True)

# Print the result
print(label_counts_all)
