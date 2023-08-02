import os
import pickle

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
all_data = pd.DataFrame(columns=['Epoch', 'Epoch RT', 'Window Avg RT', 'Label'])


def calculate_window_avg_rt(epoch_rt):
    # Calculate window average RT for each epoch: average of epoch RTs within the window
    return [np.mean(epoch_rt[max(0, i - WINDOW_SIZE): min(len(epoch_rt), i + WINDOW_SIZE)]) for i in
            range(len(epoch_rt))]


def calculate_labels(epoch_rt, window_avg_rt, alert_rt):
    # Label each epoch based on epoch RT and window average RT compared to alert RT
    return ['alert' if er < alert_rt * ALERT_RT_MULTIPLIER and wr < alert_rt * ALERT_RT_MULTIPLIER else
            'fatigued' if er > alert_rt * FATIGUE_RT_MULTIPLIER and wr > alert_rt * FATIGUE_RT_MULTIPLIER else
            'unknown' for er, wr in zip(epoch_rt, window_avg_rt)]


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

    epoch_rt_list = []
    epoch_data = {}  # New dictionary to hold EEG data and labels

    for i_epoch in range(n_epochs):
        # Indices of events that occur within this epoch
        epoch_event_indices = np.where((events[:, 0] >= i_epoch * samples_per_epoch) &
                                       (events[:, 0] < (i_epoch + 1) * samples_per_epoch))[0]

        # Ensure the indices do not go beyond the last event
        valid_indices = epoch_event_indices[epoch_event_indices < len(events) - 1]

        # Calculate epoch RT for this epoch
        epoch_rt = np.sum(events[valid_indices + 1, 0] - events[valid_indices, 0]) / eeg_data.info['sfreq']
        epoch_rt_list.append(epoch_rt)

        # Convert the EEG data to a NumPy array
        eeg_data_array = eeg_data.get_data()

        # Slice the EEG data to match this epoch
        eeg_epoch_data = eeg_data_array[:, i_epoch * samples_per_epoch: (i_epoch + 1) * samples_per_epoch]

        # Add the EEG data to the dictionary (labels will be added later)
        epoch_data[i_epoch + 1] = {'EEG Data': eeg_epoch_data}

    # Calculate alert RT: the ALERT_RT_PERCENTILE percentile of all epoch RTs
    alert_rt = np.percentile(epoch_rt_list, ALERT_RT_PERCENTILE)

    # Calculate window average RT for each epoch
    window_avg_rt_list = calculate_window_avg_rt(epoch_rt_list)

    # Label each epoch
    labels = calculate_labels(epoch_rt_list, window_avg_rt_list, alert_rt)

    # Add labels to the dictionary
    for i_epoch, label in enumerate(labels, start=1):
        epoch_data[i_epoch]['Label'] = label

    # Save the dictionary as a pickle file
    with open(f"{file_path}_labeled_data.pkl", "wb") as f:
        pickle.dump(epoch_data, f)

    # Create a DataFrame for easier viewing
    df = pd.DataFrame({
        'Epoch': np.arange(1, n_epochs + 1),
        'Epoch RT': epoch_rt_list,
        'Window Avg RT': window_avg_rt_list,
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
