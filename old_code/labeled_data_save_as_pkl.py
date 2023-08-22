# Constants
THREE_SECONDS_SAMPLES = 3 * raw.info['sfreq']

# Extract the events
events = mne.events_from_annotations(raw)[0]

# Calculate epoch RT for each epoch and extract 3 seconds of data before each event
epoch_rt_list = []
three_seconds_before_event_data = []  # To store the extracted data

for i_epoch in range(n_epochs):
    # Indices of events that occur within this epoch
    epoch_event_indices = np.where((events[:, 0] >= i_epoch * EPOCH_LENGTH * raw.info['sfreq']) &
                                   (events[:, 0] < (i_epoch + 1) * EPOCH_LENGTH * raw.info['sfreq']))[0]

    # Ensure the indices do not go beyond the last event
    valid_indices = epoch_event_indices[epoch_event_indices < len(events) - 1]

    # Calculate epoch RT for this epoch
    epoch_rt = np.sum(events[valid_indices + 1, 0] - events[valid_indices, 0]) / raw.info['sfreq']
    epoch_rt_list.append(epoch_rt)

    # Extract 3 seconds of data before each event within this epoch
    for idx in valid_indices:
        start_sample = max(0, events[idx, 0] - THREE_SECONDS_SAMPLES)
        end_sample = events[idx, 0]
        data_segment = raw.get_data(start=start_sample, stop=end_sample)
        three_seconds_before_event_data.append(data_segment)
