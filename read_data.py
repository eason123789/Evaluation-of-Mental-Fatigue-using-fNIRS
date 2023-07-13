import mne
import numpy as np

# Load the .set file using mne
eeg_data = mne.io.read_raw_eeglab('/mnt/data/s01_051017m.set', preload=True)

# Display some basic information about the dataset
eeg_data.info

# Extract events from the raw data
events, event_id = mne.events_from_annotations(eeg_data)

# Show the unique event types and their IDs
event_id


# Define the duration of the epochs (in seconds)
tmin, tmax = -0.2, 0.5  # start 200ms before the event and end 500ms after the event

# Create epochs based on the events
epochs = mne.Epochs(eeg_data, events, event_id, tmin, tmax, preload=True)

# Display the created epochs
epochs



# Initialize an empty list to store the reaction times (RTs)
rt_values = []

# Loop through the events
for i in range(len(events) - 1):
    # If the current event is a deviation onset and the next event is a response onset
    if (events[i, 2] in [1, 2]) and (events[i + 1, 2] == 3):
        # Calculate the RT (in seconds) and append it to the list
        rt = (events[i + 1, 0] - events[i, 0]) / eeg_data.info['sfreq']
        rt_values.append(rt)


# Determine the thresholds for labelling the epochs
# These are just example thresholds, you should determine the actual thresholds based on your data
optimal_threshold = np.percentile(rt_values, 33)    # 33rd percentile of the RTs
suboptimal_threshold = np.percentile(rt_values, 67) # 67th percentile of the RTs

# Initialize an empty list to store the labels
labels = []

# Loop through the RT values and label each epoch
for rt in rt_values:
    if rt <= optimal_threshold:
        labels.append('optimal')
    elif rt <= suboptimal_threshold:
        labels.append('suboptimal')
    else:
        labels.append('poor')

# Display the first few labels
labels[:10]


import numpy as np
from scipy.fft import fft

# Choose the number of frequency bins for the FFT
num_freq_bins = 100

# Initialize an empty list to store the features
features = []

# Loop through the epochs
for epoch in epochs:
    # Perform the FFT and take the absolute value of the result to get the magnitude spectrum
    spectrum = np.abs(fft(epoch))

    # Only keep the first half of the spectrum (due to symmetry)
    spectrum = spectrum[:len(spectrum)//2]

    # Resample the spectrum to have a fixed number of frequency bins
    spectrum_resampled = np.interp(np.linspace(0, len(spectrum), num_freq_bins), np.arange(len(spectrum)), spectrum)

    # Append the resampled spectrum to the list of features
    features.append(spectrum_resampled)

# Convert the list of features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Make sure the features array is 3-dimensional (required for the Transformer model)
features = np.expand_dims(features, axis=2)

# Convert the string labels to integers
label_to_int = {'optimal': 0, 'suboptimal': 1, 'poor': 2}
labels = np.array([label_to_int[label] for label in labels])

