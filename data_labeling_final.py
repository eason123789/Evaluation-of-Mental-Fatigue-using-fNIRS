import os
import mne
import numpy as np
import pandas as pd

# Constants
WINDOW_SIZE = 2  # Window size for calculating global RT
ALERT_RT_PERCENTILE = 5  # Use 5th percentile of all local RTs as alert RT
ALERT_RT_MULTIPLIER = 1.5  # Multiplier for alert RT to decide if subject is alert
FATIGUE_RT_MULTIPLIER = 3.0  # Multiplier for alert RT to decide if subject is fatigued

# Directory containing all .set files
directory = "preprocessed_data"

# DataFrame to hold results from all datasets
all_data = pd.DataFrame(columns=['Trial', 'Local RT', 'Global RT', 'Label'])

def calculate_global_rt(local_rt):
    return [np.mean(local_rt[max(0, i - WINDOW_SIZE): min(len(local_rt), i + WINDOW_SIZE)]) for i in range(len(local_rt))]

def calculate_labels(local_rt, global_rt, alert_rt):
    return ['alert' if lr < alert_rt * ALERT_RT_MULTIPLIER and gr < alert_rt * ALERT_RT_MULTIPLIER else
            'fatigued' if lr > alert_rt * FATIGUE_RT_MULTIPLIER and gr > alert_rt * FATIGUE_RT_MULTIPLIER else
            'unknown' for lr, gr in zip(local_rt, global_rt)]

# Iterate over all .set files in the main directory and its subdirectories
for subdir, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".set"):
            set_file_path = os.path.join(subdir, file)

            # Load the EEG data
            try:
                eeg_data = mne.io.read_raw_eeglab(set_file_path, preload=True)
            except Exception as e:
                print(f"Failed to load data from {set_file_path}: {e}")
                continue

            # Extract events
            events, event_id = mne.events_from_annotations(eeg_data)

            # Get indices of deviation onset events
            deviation_onset_indices = np.where((events[:, 2] == event_id['251']) | (events[:, 2] == event_id['252']))[0]

            # Filter out events that occur less than 9 seconds apart
            filtered_indices = deviation_onset_indices[
                np.where(np.diff(events[deviation_onset_indices, 0]) < 9 * eeg_data.info['sfreq'])[0]]

            # Calculate local RT for each trial: time from deviation onset to response onset
            local_rt = events[filtered_indices + 1, 0] - events[filtered_indices, 0]

            # Calculate alert RT: the ALERT_RT_PERCENTILE percentile of all local RTs
            alert_rt = np.percentile(local_rt, ALERT_RT_PERCENTILE)

            # Calculate global RT for each trial: average of local RTs within the window
            global_rt = calculate_global_rt(local_rt)

            # Label each trial based on local RT and global RT compared to alert RT
            labels = calculate_labels(local_rt, global_rt, alert_rt)

            # Create a DataFrame for easier viewing
            df = pd.DataFrame({
                'Trial': np.arange(1, len(local_rt) + 1),
                'Local RT': local_rt,
                'Global RT': global_rt,
                'Label': labels,
            })

            # Print DataFrame
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
