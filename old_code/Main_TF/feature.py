
import os
import mne
import numpy as np
from mne_features.feature_extraction import FeatureExtractor

# Set the path to the folder containing the .set files
input_folder = '/Users/easonpeng/Desktop/University of Nottingham/Evaluation of Mental Fatigue at Multilevel using functional Near Infrared Spectroscopy/Evaluation-of-Mental-Fatigue-using-fNIRS/preprocessed_data'

# Set the path to the folder where the processed data will be saved
output_folder = 'labeled_datasets_811'


# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Constants
EPOCH_LENGTH = 30
WINDOW_SIZE = 4
ALERT_RT_PERCENTILE = 10
ALERT_RT_MULTIPLIER = 1.5
FATIGUE_RT_MULTIPLIER = 2.0

def calculate_window_avg_rt(epoch_rt):
    # Calculate window average RT for each epoch: average of epoch RTs within the window
    return [np.mean(epoch_rt[max(0, i - WINDOW_SIZE): min(len(epoch_rt), i + WINDOW_SIZE)]) for i in range(len(epoch_rt))]

def calculate_labels(epoch_rt, window_avg_rt, alert_rt):
    # Label each epoch based on epoch RT and window average RT compared to alert RT
    return ['alert' if er < alert_rt * ALERT_RT_MULTIPLIER and wr < alert_rt * ALERT_RT_MULTIPLIER else
            'fatigued' for er, wr in zip(epoch_rt, window_avg_rt)]


def extract_features_with_feature_extractor(epochs):
    """
    Extract Power Spectral Density (PSD) features for each epoch using FeatureExtractor.

    Parameters:
    - epochs: mne.Epochs object containing EEG data.

    Returns:
    - psd_features: a dictionary containing PSD values for delta, theta, alpha, and beta frequency bands.
    """

    # Define feature functions and their parameters
    feature_funcs = {
        'pow_freq_bands': {
            'freq_bands': np.array([[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 40]])
        }
    }

    # Initialize the feature extractor
    extractor = FeatureExtractor(sfreq=epochs.info['sfreq'], selected_funcs=feature_funcs)

    # Get data from epochs
    data = epochs.get_data()

    # Compute features
    features = extractor.fit_transform(data)

    # Extract PSD for each band
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    psd_features = {band: feature for band, feature in zip(bands, features.T)}

    return psd_features


# Initialize a list to store the extracted features
all_features = []
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

    # Calculate epoch RT for each epoch
    epoch_rt_list = []

    for i_epoch in range(n_epochs):
        # Indices of events that occur within this epoch
        epoch_event_indices = np.where((events[:, 0] >= i_epoch * EPOCH_LENGTH * raw.info['sfreq']) &
                                       (events[:, 0] < (i_epoch + 1) * EPOCH_LENGTH * raw.info['sfreq']))[0]

        # Ensure the indices do not go beyond the last event
        valid_indices = epoch_event_indices[epoch_event_indices < len(events) - 1]

        # Calculate epoch RT for this epoch, and filter out intervals > 9 sec
        epoch_rt_values = events[valid_indices + 1, 0] - events[valid_indices, 0]
        valid_rt_values = epoch_rt_values[epoch_rt_values < 9 * raw.info['sfreq']]
        epoch_rt = np.sum(valid_rt_values) / raw.info['sfreq']
        epoch_rt_list.append(epoch_rt)

    print('epoch_rt_list:')
    print(epoch_rt_list)

    # Calculate alert RT: the ALERT_RT_PERCENTILE percentile of all epoch RTs
    alert_rt = np.percentile(epoch_rt_list, ALERT_RT_PERCENTILE)
    print('alert_rt:')
    print(alert_rt)

    # Calculate window average RT for each epoch
    window_avg_rt_list = calculate_window_avg_rt(epoch_rt_list)

    # Label each epoch
    labels = calculate_labels(epoch_rt_list, window_avg_rt_list, alert_rt)

    # Extract the epochs from the raw data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_LENGTH - 1 / raw.info['sfreq'], baseline=None, preload=True)

    # Extract features from the epochs using mne-features
    features = extract_features_with_feature_extractor(epochs)
    all_features.append(features)

    # Convert labels to numerical form for easier processing
    epoch_labels = [0 if label == 'alert' else 1 for label in labels]

    output_filepath = os.path.join(output_folder, filename.replace('.set', '.npz'))
    # Save only the processed labels and features to the output file
    np.savez_compressed(output_filepath, epoch_labels=epoch_labels, features=features)

    # Append the labels to the list
    all_labels.extend(epoch_labels)
    # Plot the distribution of the labels for the current file
    unique, counts = np.unique(epoch_labels, return_counts=True)
    print(f"File: {filename}")
    print(dict(zip(unique, counts)))

# Convert the list of all labels to a numpy array
all_labels = np.array(all_labels)
print("Overall distribution:")
unique, counts = np.unique(all_labels, return_counts=True)
print(dict(zip(unique, counts)))
