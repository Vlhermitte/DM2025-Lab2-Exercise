from typing import List, Union, Tuple

import os
from joblib import Parallel, delayed
from tqdm import tqdm
import urllib.request
import json
import sys
import zipfile
import pandas as pd
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import Configurator

# ---------------------------
# Neural Dataset
# ---------------------------

class NeuralDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - 'neural_features': np.ndarray shape (T, 512)  (dtype float or similar)
      - 'transcriptions': str (ASCII) or List[int] in 0..127
      - 'sentence_label': str (used to determine target length)
    """
    def __init__(self, df: pd.DataFrame, char_to_id: dict = None, augment=False, smoothing: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.smoothing = smoothing
        self.char_to_id = char_to_id

        # Apply smoothing preemptively if needed
        if self.smoothing:
            for i in range(len(self.df)):
                self.df.at[i, 'neural_features'] = gaussian_filter1d(self.df.at[i, 'neural_features'], sigma=1, axis=0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x_arr = self.df.iloc[idx]['neural_features']   # (T, 512)
        x = torch.tensor(x_arr, dtype=torch.float32)   # (T, 512)

        # Z-Score Normalization + Clipping
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-5)
        x = torch.clamp(x, min=-5.0, max=5.0)

        # Get Raw ASCII sequence
        raw_y = self.df.iloc[idx]['transcriptions']

        # --- FIX: Convert Raw ASCII to Dense IDs ---
        if self.char_to_id is not None:
            # Filter out original 0s (padding) if present in raw data
            # Map ASCII -> Dense ID
            # Assuming raw_y is a list of ints or numpy array of ints
            dense_y = []
            for item in raw_y:
                val = item.item() if isinstance(item, (np.generic, torch.Tensor)) else item
                if val == 0: continue  # Skip old padding/blanks
                if val in self.char_to_id:
                    dense_y.append(self.char_to_id[val])
                else:
                    # Handle unknown chars? Map to blank or special UNK if you had one
                    pass
            y = torch.tensor(dense_y, dtype=torch.long)
        else:
            # Fallback (legacy behavior)
            y = torch.tensor(raw_y, dtype=torch.long)

        if self.augment:
            x = apply_augmentation(x)

        # remove padding from y. Take only the first len(self.df.iloc[idx]['sentence_label']) elements
        # len_sentence = len(self.df.iloc[idx]['sentence_label'])
        # y = y[:len_sentence]

        if torch.any(y == 0):
            raise ValueError(f"Found PAD_ID (0) in target sequence at index {idx}. Check data preprocessing.")

        return x, y

def collate_transducer(batch, pad_id=0, batch_first=False):
    xs, ys = zip(*batch)
    x_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    y_lengths = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)

    xs_padded = pad_sequence(xs, batch_first=batch_first)
    ys_padded = pad_sequence(ys, batch_first=batch_first, padding_value=pad_id)

    return xs_padded, x_lengths, ys_padded, y_lengths

# ---------------------------
# Data Augmentation
# ---------------------------

def apply_augmentation(x):
    """
    Custom Augmentation for 4-Array Speech BCI Data.
    Layout:
        0-255: TCs (4 arrays x 64 elecs)
        256-511: SBP (4 arrays x 64 elecs)
    """
    T, F = x.shape

    # 1. Gaussian Noise (Standard)
    # Adds robustness to thermal noise/background activity
    if np.random.rand() < 0.5:
        noise_level = 0.1
        x = x + torch.randn_like(x) * noise_level

    # 2. Constant Offset (DC Shift)
    # Simulates the electrode voltage baseline shifting
    if np.random.rand() < 0.5:
        offset = torch.randn(1, x.shape[1]) * 0.2  # 0.2 matches baseline args
        x = x + offset

    # 3. Random Walk (Slow Drift)
    # Simulates gradual impedance changes over time
    if np.random.rand() < 0.5:
        drift = torch.randn_like(x) * 0.05  # Small steps
        drift = torch.cumsum(drift, dim=0)  # Accumulate over time
        x = x + drift

    # 4. Array Dropout (Simulates hardware failure of a specific pedestal)
    # We have 4 arrays. Each controls 64 indices in the first half
    # and 64 indices in the second half.
    if np.random.rand() < 0.2:  # 20% chance to drop an entire array
        array_idx = np.random.randint(0, 4)  # 0, 1, 2, or 3

        # Calculate indices
        tc_start = array_idx * 64
        tc_end = (array_idx + 1) * 64
        sbp_start = 256 + (array_idx * 64)
        sbp_end = 256 + ((array_idx + 1) * 64)

        # Zero out both TC and SBP for this array
        x[:, tc_start:tc_end] = 0.0
        x[:, sbp_start:sbp_end] = 0.0

    # 5. Modality Dropout (Simulates processing artifact)
    # Occasionally drop ALL Threshold Crossings or ALL SBP
    if np.random.rand() < 0.1:  # 10% chance
        if np.random.rand() < 0.5:
            x[:, 0:256] = 0.0  # Drop all TCs
        else:
            x[:, 256:512] = 0.0  # Drop all SBPs

    # 6. Standard Time Masking (Simulates loss of attention/artifacts)
    if np.random.rand() < 0.5:
        num_masks = 2
        max_mask_len = max(1, T // 10)
        for _ in range(num_masks):
            mask_len = np.random.randint(0, max_mask_len)
            t0 = np.random.randint(0, T - mask_len)
            x[t0:t0 + mask_len, :] = 0.0

    return x

# ---------------------------
# Utility Functions
# ---------------------------

def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data

def load_and_process(file_path):
    """
    Helper function to be run by a worker process.
    Loads the file and returns the DataFrame.
    """
    data = load_h5py_file(file_path)
    return pd.DataFrame(data)


def get_dataframes(path, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folders = os.listdir(path)
    train_files = []
    val_files = []

    # 1. Collect file paths (Fast, no changes needed)
    for folder in folders:
        if folder.startswith("."):
            continue

        folder_path = os.path.join(path, folder)
        # Check if it's actually a directory to avoid errors
        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)
        for file in files:
            full_path = os.path.join(folder_path, file)
            if file.endswith("train.hdf5"):
                train_files.append(full_path)
            elif file.endswith("val.hdf5"):
                val_files.append(full_path)

    # 2. Handle Debug Logic (Pre-slice the list)
    if debug:
        # Load only a limited number of files
        train_files = train_files[:1]  # Adjusted to 2 or 4 based on your preference
        val_files = val_files[:1]

    # 3. Parallel Execution
    # n_jobs=-1 uses all available CPU cores.
    # We use a list comprehension to gather the Delayed objects.

    print(f"Loading {len(train_files)} train files...")
    train_dfs_list = Parallel(n_jobs=1)(
        delayed(load_and_process)(f) for f in tqdm(train_files, desc="Processing Train")
    )

    print(f"Loading {len(val_files)} val files...")
    val_dfs_list = Parallel(n_jobs=1)(
        delayed(load_and_process)(f) for f in tqdm(val_files, desc="Processing Val")
    )

    # 4. Single Concatenation (Much faster than appending in a loop)
    # If the list is empty, return an empty DF to avoid errors
    train_df = pd.concat(train_dfs_list, ignore_index=True) if train_dfs_list else pd.DataFrame()
    val_df = pd.concat(val_dfs_list, ignore_index=True) if val_dfs_list else pd.DataFrame()

    return train_df, val_df


def display_progress_bar(block_num, block_size, total_size, message=""):
    """"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()

def download_data(DATA_DIR: str = "brain-to-text-25/"):
    DRYAD_DOI = "10.5061/dryad.dncjsxm85"

    ## Make sure the command is being run from the right place and we can see the data/
    ## directory.

    data_dirpath = os.path.abspath(DATA_DIR)
    if not os.path.exists(data_dirpath):
        os.makedirs(data_dirpath)

    assert os.getcwd().endswith(
        "BrainToText2025"
    ), f"Please run the download command from the BrainToText2025 directory (instead of {os.getcwd()})"
    assert os.path.exists(
        data_dirpath
    ), "Cannot find the data directory to download into."

    ## Get the list of files from the latest version on Dryad.

    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"][
        "stash:files"
    ]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    ## Download each file into the data directory (and unzip for certain files).

    for file_info in file_infos:
        filename = file_info["path"]

        if filename == "README.md":
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(data_dirpath, filename)

        urllib.request.urlretrieve(
            download_url,
            download_to_filepath,
            reporthook=lambda *args: display_progress_bar(
                *args, message=f"Downloading {filename}"
            ),
        )
        sys.stdout.write("\n")

        # If this file is a zip file, unzip it into the data directory.

        if file_info["mimeType"] == "application/zip":
            print(f"Extracting files from {filename} ...")
            with zipfile.ZipFile(download_to_filepath, "r") as zf:
                zf.extractall(data_dirpath)

    print(f"\nDownload complete. See data files in {data_dirpath}\n")