import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne_features.feature_extraction import extract_features

# Set the path to the folder containing the .set files
input_folder = '../preprocessed_data'

# Set the path to the folder where the processed data will be saved
output_folder = '../labeled_data_822'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a list to hold the labels of all epochs
all_labels = []
all_spectral_data = []
all_temporal_data = []
# Constants
EPOCH_LENGTH = 30
WINDOW_SIZE = 2
ALERT_RT_PERCENTILE = 5
ALERT_RT_MULTIPLIER = 1.5
FATIGUE_RT_MULTIPLIER = 2.5


def zscore_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def positional_encoding(position, d_model):
    angle_rads = np.arange(d_model)[np.newaxis, :] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding

def calculate_window_avg_rt(epoch_rt):
    # Calculate window average RT for each epoch: average of epoch RTs within the window
    return [np.mean(epoch_rt[max(0, i - WINDOW_SIZE): min(len(epoch_rt), i + WINDOW_SIZE)]) for i in range(len(epoch_rt))]

def calculate_labels(epoch_rt, window_avg_rt, alert_rt):
    # Label each epoch based on epoch RT and window average RT compared to alert RT
    return ['alert' if er < alert_rt * ALERT_RT_MULTIPLIER and wr < alert_rt * ALERT_RT_MULTIPLIER else
            'fatigued' for er, wr in zip(epoch_rt, window_avg_rt)]

# Iterate over the folders in the input folder
for foldername in os.listdir(input_folder):
    if foldername == '.DS_Store':
        continue
    # Get the .set file path
    filename = foldername
    filepath = os.path.join(input_folder, foldername, filename)

    # Load the data
    raw = mne.io.read_raw_eeglab(filepath, preload=True)

    # Set the epoch duration (in seconds)
    epoch_duration = 30

    # Calculate the number of epochs
    n_epochs = int(np.ceil(raw.times[-1] / epoch_duration))

    # Extract the events
    events = mne.events_from_annotations(raw)[0]

    # Calculate epoch RT for each epoch
    epoch_rt_list = []
    for i_epoch in range(n_epochs):
        # Indices of events that occur within this epoch
        epoch_event_indices = np.where((events[:, 0] >= i_epoch * EPOCH_LENGTH * raw.info['sfreq']) &
                                       (events[:, 0] < (i_epoch + 1) * EPOCH_LENGTH * raw.info['sfreq']))[0]

        # Ensure the indices do not go beyond the last event
        valid_indices = epoch_event_indices[epoch_event_indices < len(events) - 1]

        # Calculate epoch RT for this epoch
        epoch_rt = np.sum(events[valid_indices + 1, 0] - events[valid_indices, 0]) / raw.info['sfreq']
        epoch_rt_list.append(epoch_rt)

    # print('epoch_rt_list:')
    # print(epoch_rt_list)

    # Calculate alert RT: the ALERT_RT_PERCENTILE percentile of all epoch RTs
    alert_rt = np.percentile(epoch_rt_list, ALERT_RT_PERCENTILE)
    # print('alert_rt:')
    # print(alert_rt)
    # Calculate window average RT for each epoch
    window_avg_rt_list = calculate_window_avg_rt(epoch_rt_list)

    # Label each epoch
    labels = calculate_labels(epoch_rt_list, window_avg_rt_list, alert_rt)

    # Convert labels to numerical form for easier processing
    epoch_labels = [0 if label == 'alert' else 1 for label in labels]

    # Get the indices of the specified channels
    channels = ['FCZ', 'TP7', 'CP3', 'OZ']
    channel_indices = [raw.ch_names.index(ch) for ch in channels]

    # Initialize lists to hold the spectral and temporal data
    spectral_data = []
    temporal_data = []

    # Iterate over the epochs
    for i in range(n_epochs):
        # Extract the entire epoch data
        start = i * epoch_duration
        end = min((i + 1) * epoch_duration, raw.times[-1])
        # start = max(i * epoch_duration, 0)
        # end = min(i * epoch_duration + 9, raw.times[-1])
        epoch_data = raw.copy().crop(tmin=start, tmax=end).get_data()

        freqs = np.fft.rfftfreq(epoch_data.shape[1], 1 / raw.info['sfreq'])

        fft_vals = np.abs(np.fft.rfft(epoch_data, axis=1))
        delta_band = np.logical_and(freqs >= 0.5, freqs <= 4.5)
        theta_band = np.logical_and(freqs >= 4.5, freqs <= 8.5)
        alpha_band = np.logical_and(freqs >= 8.5, freqs <= 11.5)
        sigma_band = np.logical_and(freqs >= 11.5, freqs <= 15.5)
        beta_band = np.logical_and(freqs >= 15.5, freqs <= 30)
        delta_band = np.mean(fft_vals[:, delta_band], axis=-1)
        theta_data = np.mean(fft_vals[:, theta_band], axis=-1)
        alpha_data = np.mean(fft_vals[:, alpha_band], axis=-1)
        sigma_band = np.mean(fft_vals[:, sigma_band], axis=-1)
        beta_data = np.mean(fft_vals[:, beta_band], axis=-1)



        # Append the spectral data to the list
        spectral_data.append(np.stack([delta_band, theta_data, alpha_data, sigma_band, beta_data]))
        temp_data = epoch_data[channel_indices, :]
        #temporal_data.append(temp_data)

        # # Only append data of shape (4, 4501) to temporal_data
        if temp_data.shape == (4,15001):
            temporal_data.append(temp_data)
        else:
            print(f"Skipping epoch {i} due to shape {temp_data.shape}")

        # Convert the lists to numpy arrays
    spectral_data = np.array(spectral_data)

    temporal_data = np.array(temporal_data)

    spectral_data = zscore_normalization(spectral_data)
    temporal_data = zscore_normalization(temporal_data)

    all_spectral_data.append(spectral_data)
    all_temporal_data.append(temporal_data)



    # Append the labels to the list
    all_labels.extend(epoch_labels)
    # Plot the distribution of the labels for the current file
    unique, counts = np.unique(epoch_labels, return_counts=True)
    print(f"File: {filename}")
    print(dict(zip(unique, counts)))

all_spectral_data = np.concatenate(all_spectral_data, axis=0)
all_temporal_data = np.concatenate(all_temporal_data, axis=0)
 # Add positional encoding to all_spectral_data
d_model_spectral = all_spectral_data.shape[-1]
pos_encoding_spectral = positional_encoding(all_spectral_data.shape[0], d_model_spectral)
all_spectral_data += pos_encoding_spectral

# Add positional encoding to all_temporal_data
d_model_temporal = all_temporal_data.shape[-1]
pos_encoding_temporal = positional_encoding(all_temporal_data.shape[0], d_model_temporal)
all_temporal_data += pos_encoding_temporal

# Now, save the concatenated data with positional encoding
output_filepath_all = os.path.join(output_folder, 'all_data.npz')
np.savez(output_filepath_all, spectral_data=all_spectral_data, temporal_data=all_temporal_data, labels=all_labels)
# Convert the list of all labels to a numpy array
all_labels = np.array(all_labels)
print("Overall distribution:")
unique, counts = np.unique(all_labels, return_counts=True)
print(dict(zip(unique, counts)))
# Plot the distribution of the labels
plt.hist(all_labels, bins=[0, 1])
plt.xticks([0.5, 1.5], ['alert', 'fatigued'])
plt.show()
