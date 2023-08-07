import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the folder containing the .set files
input_folder = 'preprocessed_data'

# Set the path to the folder where the processed data will be saved
output_folder = 'data_labeling_R'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a list to hold the labels of all epochs
all_labels = []

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

    # Get the indices of the events of type 251 or 252 (deviation onset) and 253 (response onset)
    deviation_onset_indices = np.where((events[:, 2] == 251) | (events[:, 2] == 252))[0]
    response_onset_indices = np.where(events[:, 2] == 253)[0]

    # Calculate the reaction times
    reaction_times = events[response_onset_indices, 0] - events[deviation_onset_indices, 0]

    # Convert the reaction times into epochs
    reaction_time_epochs = np.floor(events[deviation_onset_indices, 0] / raw.info['sfreq'] / epoch_duration).astype(int)

    epoch_rt_list = []
    # Calculate the local reaction time for each epoch
    reaction_times_per_epoch = [[] for _ in range(n_epochs)]
    for rt, epoch in zip(reaction_times, reaction_time_epochs):
        if epoch < n_epochs:  # ignore events that fall into the discarded last epoch
            reaction_times_per_epoch[epoch].append(rt)
    epochs_local_rt = [np.sum(rt) if rt else 0 for rt in reaction_times_per_epoch]
    # Calculate the global reaction time for each epoch
    n = 1  # window size for calculating globalRT
    epochs_global_rt = np.convolve(epochs_local_rt, np.ones(2*n+1), mode='same') / (2*n+1)
    epoch_rt_list.append(epochs_local_rt)
    print('epoch_rt_list:')
    print(epoch_rt_list)
    # Calculate the alert reaction time
    alert_rt = np.percentile(epoch_rt_list, 10)
    print('alert_rt:')
    print(alert_rt)

    # Calculate the labels for each epoch
    epoch_labels = np.zeros(n_epochs)
    for i in range(n_epochs):
        if epochs_local_rt[i] < 2.0 * alert_rt and epochs_global_rt[i] < 2.0 * alert_rt:
            epoch_labels[i] = 0  # alert
        elif epochs_local_rt[i] > 1.5 * alert_rt and epochs_global_rt[i] > 1.5 * alert_rt:
            epoch_labels[i] = 1  # drowsy

    # Get the indices of the specified channels
    channels = ['FCZ', 'TP7', 'CP3', 'OZ']
    channel_indices = [raw.ch_names.index(ch) for ch in channels]

    # Initialize lists to hold the spectral and temporal data
    spectral_data = []
    temporal_data = []

    # Iterate over the epochs
    for i in range(n_epochs):
        # Extract the 3 seconds of data before the end of the epoch
        start = min((i + 1) * epoch_duration - 3, raw.times[-1] - 3)
        end = min((i + 1) * epoch_duration, raw.times[-1])
        epoch_data = raw.copy().crop(tmin=start, tmax=end).get_data()

        # Perform FFT and extract the theta, alpha, and beta bands
        freqs = np.fft.rfftfreq(int(raw.info['sfreq'] * 3), 1/raw.info['sfreq'])
        fft_vals = np.abs(np.fft.rfft(epoch_data, axis=1))
        theta_band = np.logical_and(freqs >= 4, freqs <= 8)
        alpha_band = np.logical_and(freqs >= 8, freqs <= 13)
        beta_band = np.logical_and(freqs >= 13, freqs <= 30)
        theta_data = np.mean(fft_vals[:, theta_band], axis=-1)
        alpha_data = np.mean(fft_vals[:, alpha_band], axis=-1)
        beta_data = np.mean(fft_vals[:, beta_band], axis=-1)

        # Append the spectral data to the list
        spectral_data.append(np.stack([theta_data, alpha_data, beta_data]))

        # Extract the temporal data from the specified channels
        temporal_data.append(epoch_data[channel_indices, :])

    # Convert the lists to numpy arrays
    spectral_data = np.array(spectral_data)
    temporal_data = np.array(temporal_data)

    # Save the spectral data, temporal data, and labels to a numpy file
    output_filepath = os.path.join(output_folder, filename.replace('.set', '.npz'))
    np.savez(output_filepath, spectral_data=spectral_data, temporal_data=temporal_data, labels=epoch_labels)

    # Append the labels to the list
    all_labels.extend(epoch_labels)
    # Plot the distribution of the labels for the current file
    unique, counts = np.unique(epoch_labels, return_counts=True)
    print(f"File: {filename}")
    print(dict(zip(unique, counts)))

# Convert the list of all labels to a numpy array
all_labels = np.array(all_labels)

# Plot the distribution of the labels
plt.hist(all_labels, bins=[0, 1, 2])
plt.xticks([0.5, 1.5], ['alert', 'drowsy'])
plt.show()
