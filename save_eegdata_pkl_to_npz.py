import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Get a list of all subdirectories
subdirectories = [d for d in os.listdir('preprocessed_data') if os.path.isdir(os.path.join('preprocessed_data', d))]

# Initialize a label encoder
le = LabelEncoder()

# Process each file in each subdirectory
for i, subdir in enumerate(subdirectories):
    file_path = os.path.join('preprocessed_data', subdir, subdir + '_labeled_data.pkl')
    with open(file_path, 'rb') as f:
        data = pd.read_pickle(f)
    # Convert each EEG data into a sequence and add its label
    sequences = []
    labels = []
    for key in data:
        sequence = data[key]['EEG Data'][0]
        label = data[key]['Label']
        sequences.append(sequence)
        labels.append(label)
    # Encode the labels
    labels = le.fit_transform(labels)
    # Save the sequences and labels to a .npz file
    np.savez('labeled_data/processed_data_{}.npz'.format(i), sequences=sequences, labels=labels)

