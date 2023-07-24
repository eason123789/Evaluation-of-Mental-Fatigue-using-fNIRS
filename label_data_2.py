import mne
import numpy as np
import pandas as pd


def extract_events_and_epochs(eeg_data, tmin=0, tmax=30):
    # Extract events from the raw data and create epochs based on the events.
    events, event_id = mne.events_from_annotations(eeg_data)
    epochs = mne.Epochs(eeg_data, events, event_id, tmin, tmax, preload=False, baseline=None)
    return events, epochs


# Load the EEG dataset
eeg_dataset = mne.io.read_raw_eeglab('preprocessed_data/s01_060926_1n.set/s01_060926_1n.set', preload=True)

# Extract events and create epochs
events, epochs = extract_events_and_epochs(eeg_dataset, tmin=0, tmax=30)

# Define the event codes
deviation_onset_codes = [1, 2]  # 1 and 2 correspond to '251' and '252'
response_onset_code = 3  # 3 corresponds to '253'

# Extract the event times and types
event_times = epochs.events[:, 0]
event_types = epochs.events[:, 2]

# Get the deviation onset times
deviation_onset_times = event_times[np.isin(event_types, deviation_onset_codes)]

# Compute the reaction times (RTs) for each response onset
reaction_times = []
response_onset_times = event_times[event_types == response_onset_code]

for response_onset_time in response_onset_times:
    # Find the most recent deviation onset for this response onset
    recent_deviation_onset_time = deviation_onset_times[deviation_onset_times < response_onset_time]
    if len(recent_deviation_onset_time) > 0:
        # Compute the RT in seconds
        rt = (response_onset_time - recent_deviation_onset_time[-1]) / eeg_dataset.info['sfreq']
        reaction_times.append(rt)

reaction_times = np.array(reaction_times)
# print(reaction_times)
# Compute the local RT (the reaction time of the current event) and the global RT (the average reaction time of all
# events within the 90 s prior to the current event)
local_rts = reaction_times
global_rts = np.array([np.mean(reaction_times[max(0, i - 90):i + 1]) for i in range(len(reaction_times))])


# Compute the alert RT (5% of the minimum local RT)
alert_rt = np.percentile(local_rts, 5)


print(alert_rt)
# Classify each event as "awake", "fatigued", or "unknown" based on the local RT and global RT
states = []
for local_rt, global_rt in zip(local_rts, global_rts):
    if local_rt < 1.5 * alert_rt and global_rt < 1.5 * alert_rt:
        state = "awake"
    elif local_rt > 2.5 * alert_rt and global_rt > 2.5 * alert_rt:
        state = "fatigued"
    else:
        state = "unknown"
    states.append(state)

# Recompute the fatigue states and store them in a dictionary
fatigue_states = {time: state for time, state in zip(response_onset_times, states)}

# Create a metadata DataFrame to hold the fatigue states
metadata = pd.DataFrame(index=range(len(epochs.events)), columns=['Fatigue State'])

# Assign the fatigue state to each epoch
for i, event in enumerate(epochs.events):
    # Get the time of the event associated with this epoch
    event_time = event[0]

    # Find the closest response onset time
    closest_response_onset_time = min(response_onset_times, key=lambda x: abs(x - event_time))

    # Get the fatigue state associated with this response onset time
    fatigue_state = fatigue_states[closest_response_onset_time]

    # Assign the fatigue state to the epoch
    metadata.loc[i, 'Fatigue State'] = fatigue_state

# Add the metadata to the epochs
epochs.metadata = metadata

print(epochs.metadata)

# Compute the proportions of each fatigue state
fatigue_state_counts = epochs.metadata['Fatigue State'].value_counts()
fatigue_state_proportions = fatigue_state_counts / len(epochs.metadata)

print(fatigue_state_proportions)
