import numpy as np
from scipy.fft import fft
import mne

def load_data(file_path):
    """
    Load EEG data from a .set file.
    """
    eeg_data = mne.io.read_raw_eeglab(file_path, preload=True)
    return eeg_data

def extract_events_and_epochs(eeg_data, tmin=-0.2, tmax=0.5):
    """
    Extract events from the raw data and create epochs based on the events.
    """
    events, event_id = mne.events_from_annotations(eeg_data)
    epochs = mne.Epochs(eeg_data, events, event_id, tmin, tmax, preload=True)
    return events, epochs

def calculate_reaction_times(events, eeg_data):
    """
    Calculate the reaction times (RTs) based on the events.
    """
    rt_values = []
    for i in range(len(events) - 1):
        if (events[i, 2] in [1, 2]) and (events[i + 1, 2] == 3):
            rt = (events[i + 1, 0] - events[i, 0]) / eeg_data.info['sfreq']
            rt_values.append(rt)
    return rt_values

def label_epochs(rt_values, optimal_percentile=33, suboptimal_percentile=67):
    """
    Label the epochs based on the reaction times (RTs).
    """
    optimal_threshold = np.percentile(rt_values, optimal_percentile)
    suboptimal_threshold = np.percentile(rt_values, suboptimal_percentile)
    labels = []
    for rt in rt_values:
        if rt <= optimal_threshold:
            labels.append('optimal')
        elif rt <= suboptimal_threshold:
            labels.append('suboptimal')
        else:
            labels.append('poor')
    return labels

# def extract_features(epochs, num_freq_bins=100):
#     """
#     Extract features from the epochs using the Fast Fourier Transform (FFT).
#     """
#     features = []
#     for epoch in epochs:
#         spectrum = np.abs(fft(epoch))
#         print("Spectrum shape:", spectrum.shape)  # Add this line to check the shape of spectrum
#         spectrum = spectrum[:len(spectrum)//2]
#         spectrum_resampled = np.interp(np.linspace(0, len(spectrum), num_freq_bins), np.arange(len(spectrum)), spectrum)
#         features.append(spectrum_resampled)
#     return np.array(features)

def extract_features(epochs, num_freq_bins=100):
    """
    Extract features from the epochs using the Fast Fourier Transform (FFT).
    """
    features = []
    for epoch in epochs:
        feature_vector = []
        for channel in epoch:
            spectrum = np.abs(fft(channel))
            spectrum = spectrum[:len(spectrum)//2]
            spectrum_resampled = np.interp(np.linspace(0, len(spectrum), num_freq_bins), np.arange(len(spectrum)), spectrum)
            feature_vector.extend(spectrum_resampled)
        features.append(feature_vector)
    return np.array(features)


def convert_labels_to_integers(labels):
    """
    Convert the string labels to integers.
    """
    label_to_int = {'optimal': 0, 'suboptimal': 1, 'poor': 2}
    return np.array([label_to_int[label] for label in labels])

def get_data():
    file_path = 'preprocessed_data/s01_051017m.set/s01_051017m.set'
    eeg_data = load_data(file_path)
    events, epochs = extract_events_and_epochs(eeg_data)
    rt_values = calculate_reaction_times(events, eeg_data)
    labels = label_epochs(rt_values)
    features = extract_features(epochs)
    labels = convert_labels_to_integers(labels)
    return features, labels

def main():
    file_path = 'preprocessed_data/s01_051017m.set/s01_051017m.set'
    eeg_data = load_data(file_path)
    events, epochs = extract_events_and_epochs(eeg_data)
    rt_values = calculate_reaction_times(events, eeg_data)
    labels = label_epochs(rt_values)
    features = extract_features(epochs)
    labels = convert_labels_to_integers(labels)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

if __name__ == "__main__":
    main()
