import mne
import numpy as np

# Load the EEG dataset
eeg_dataset = mne.io.read_raw_eeglab('preprocessed_data/s01_051017m.set/s01_051017m.set', preload=True)

# Extract the events from the dataset
events = mne.events_from_annotations(eeg_dataset)

# Define the event codes
deviation_onset_codes = [1, 2]  # 1 and 2 correspond to '251' and '252'
response_onset_code = 3  # 3 corresponds to '253'

# Extract the event times and types
event_times, _, event_types = events[0].T

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

# Compute the local RT (the reaction time of the current event) and the global RT (the average reaction time of all
# events within the 90 s prior to the current event)
local_rts = reaction_times
global_rts = np.array([np.mean(reaction_times[max(0, i-90):i+1]) for i in range(len(reaction_times))])

# # Compute the alert RT (5% of the minimum local RT)
# alert_rt = np.min(local_rts) * 0.05

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

# Print the first few reaction times and states
print(reaction_times[:5])
print(local_rts[:5])
print(global_rts[:5])
print(states[:5])
