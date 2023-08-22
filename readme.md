
## How to Use

### Data Preprocessing

1. Open `process_eeg.py` and change the path for reading `.set` files.
2. Run `process_eeg.py`. This will process all `.set` files and save them as `all_data.npz`.

### Training the Transformer Model

1. Run `eeg_transformer.py`. This will use `all_data.npz` for training and save the best model.

### Training the CNN Model

1. Run `cnn.py`. This will use `all_data.npz` for training, for comparison.

## Note

- Make sure to run `process_eeg.py` before running `eeg_transformer.py` or `cnn.py`.
