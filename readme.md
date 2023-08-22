
```markdown
# Final Code for EEG Data Processing and Analysis

This repository contains code for processing and analyzing EEG data. It includes utilities for data preprocessing, Transformer model training, and CNN model training for comparison.

## Folder Structure

```plaintext
final_code/
|-- process_eeg.py: For preprocessing EEG data
|-- eeg_transformer.py: For training a Transformer model
|-- cnn.py: For training a CNN model for comparison
preprocessed_data/: Folder for storing preprocessed data
```

## Preprocessed Data

The `preprocessed_data` folder is meant to store source EEG data files. While this folder does not contain the complete dataset, it serves as a guideline for how your preprocessed data should be organized. 

```plaintext
preprocessed_data/
    s01_051017m.set/
        |-- s01_051017m.fdt
        |-- s01_051017m.set
```

## How to Use

### Data Preprocessing

1. Open `process_eeg.py` and modify the path to read `.set` files for EEG data.
2. Run the script `process_eeg.py`. This will process all the `.set` files and save the preprocessed data as `all_data.npz` in the `preprocessed_data` folder.

### Training the Transformer Model

1. Execute `eeg_transformer.py`. This script will use the `all_data.npz` file stored in the `preprocessed_data` folder for training and will save the best-performing model.

### Training the CNN Model

1. Execute `cnn.py`. This script will also use the `all_data.npz` file from the `preprocessed_data` folder for training. This is intended for comparison purposes with the Transformer model.

## Notes

- Make sure to execute `process_eeg.py` before running either `eeg_transformer.py` or `cnn.py` to ensure that the data is properly preprocessed.
```
