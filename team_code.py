#!/usr/bin/env python
# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader, TensorDataset
from helper_code import *
from helper_code import get_source
from scipy.signal import resample
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import random
import numpy as np
from scipy.signal import welch
import neurokit2 as nk
import pickle
import multiprocessing
from functools import partial
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import neurokit2 as nk
from helper_code import load_signals, load_label, load_header, get_source
from transformers import get_cosine_schedule_with_warmup

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)


seed_everything(42)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def ranking_hinge_loss(scores, target, margin=1.0, num_pairs=1000):
    # scores: (B,) = model’s score for “class 1”
    pos_idx = torch.nonzero(target==1, as_tuple=False).view(-1)
    neg_idx = torch.nonzero(target==0, as_tuple=False).view(-1)
    if len(pos_idx)==0 or len(neg_idx)==0:
        return torch.tensor(0., device=scores.device)

    # sample a subset of pos/neg for efficiency
    p = pos_idx[torch.randint(len(pos_idx), (min(len(pos_idx),num_pairs),))]
    n = neg_idx[torch.randint(len(neg_idx), (min(len(neg_idx),num_pairs),))]
    s_pos = scores[p].unsqueeze(1)  # (P,1)
    s_neg = scores[n].unsqueeze(0)  # (1,N)
    # want s_pos - s_neg ≥ margin → loss when < margin
    losses = torch.relu(margin - (s_pos - s_neg))
    return losses.mean()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
LEARNING_RATE = 0.0001*8 # # it was 0.001 for transformer
SPATIAL_INPUT_DIM = 10
TEMPORAL_INPUT_DIM = 120
N_HEADS = 1
N_LAYERS = 1
DROPOUT = 0.2
NUM_CLASSES = 2  


def filter_data(signal, lowcut=0.5, highcut=40.0, fs=250.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


class conv_model(nn.Module):
    def __init__(self, input_dim=6, num_leads=12, num_classes=2):
        super(conv_model, self).__init__()
        self.input_dim = input_dim
        self.num_leads = num_leads
        self.num_classes = num_classes
        
        self.conv1_layer = nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1)
        self.conv2_layer = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1)
        
        #make conv1 as a sequential layer
        self.conv1 = nn.Sequential(
            self.conv1_layer,
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
     
        )
        self.conv2 = nn.Sequential(
            self.conv2_layer,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1)),  # Downsample
            nn.Dropout(0.2)
        )

        # Fully connected layers
        self.layer_norm = nn.LayerNorm(3072)  # Adjust input size based on conv output
        self.fc1 = nn.Linear(3072, 128)  # Adjust input size based on conv output
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.7)
        self.flatten = nn.Flatten()

        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, num_leads, input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)  # Flatten the output for fully connected layers
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

#         return x
def normalize_signal(signal):
    """
    Normalizes a signal to the range [-1, 1] using min-max scaling.

    Args:
        signal (np.ndarray): The input signal array.

    Returns:
        np.ndarray: The normalized signal.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val > 0:
        # Formula: -1 + 2 * (x - min) / (max - min)
        return -1 + 2 * (signal - min_val) / (max_val - min_val)
    else:
        # Handle the case of a flat signal to avoid division by zero
        return np.zeros_like(signal)




def process_single_lead(lead_data, lead_idx, original_fs, desired_sampling_rate, expected_length, feature_names):
    """Process a single lead and extract all requested features including P wave features."""
    ecg_signal = lead_data
    lead_features = {name: np.nan for name in feature_names}
    
    try:
        # 1. Fast normalization
        normalized_signal = normalize_signal(ecg_signal)
        
        # 2. Faster resampling with exact length
        resampled_signal = nk.signal_resample(
            normalized_signal,
            sampling_rate=original_fs,
            desired_sampling_rate=desired_sampling_rate,
            desired_length=expected_length,
            
        )
        
        # 3. Use faster cleaning method - pantompkins is faster than neurokit
        cleaned_signal = nk.ecg_clean(
            resampled_signal, 
            sampling_rate=desired_sampling_rate, 
            method="pantompkins"
        )
        
        # 4. Faster R-peak detection
        _, info = nk.ecg_peaks(
            cleaned_signal, 
            sampling_rate=desired_sampling_rate,
            method="pantompkins"
        )
        r_peaks = info['ECG_R_Peaks']
        
        if len(r_peaks) < 2:
            return lead_idx, lead_features
            
        # Calculate RR Interval Variability (Heart Rate Variability)
        if len(r_peaks) >= 3:  # Need at least 3 R-peaks for meaningful HRV
            rr_intervals = np.diff(r_peaks) / desired_sampling_rate  # Convert to seconds
            # Filter physiologically reasonable RR intervals (0.3s to 2.0s)
            valid_rr = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
            
            if len(valid_rr) >= 2:
                # RMSSD - Root Mean Square of Successive Differences (ms)
                rr_diff = np.diff(valid_rr * 1000)  # Convert to ms
                lead_features['RR_Interval_RMSSD_ms'] = np.sqrt(np.mean(rr_diff**2))
            else:
                lead_features['RR_Interval_RMSSD_ms'] = np.nan
        else:
            lead_features['RR_Interval_RMSSD_ms'] = np.nan
            
        # 5. Delineate waveforms
        try:
            _, waves_info = nk.ecg_delineate(
                cleaned_signal, 
                r_peaks, 
                sampling_rate=desired_sampling_rate, 
                method="dwt"
            )
            
            # 6. Calculate all the requested features
            # QRS Duration features
            if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info:
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                
                # Calculate QRS durations directly, handling NaNs
                qrs_durations = (s_peaks - q_peaks) / desired_sampling_rate
                # Filter valid values
                valid_qrs = ~np.isnan(qrs_durations) & (qrs_durations > 0)
                if np.any(valid_qrs):
                    valid_qrs_durations = qrs_durations[valid_qrs]
                    lead_features['Mean_QRS_Duration_ms'] = np.nanmean(valid_qrs_durations) * 1000
                    lead_features['Std_QRS_Duration_ms'] = np.nanstd(valid_qrs_durations) * 1000 if len(valid_qrs_durations) > 1 else 0
            
            # QT Interval features
            if 'ECG_T_Offsets' in waves_info and 'ECG_Q_Peaks' in waves_info:
                t_offsets = np.array(waves_info['ECG_T_Offsets'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                
                # Calculate QT intervals directly, handling NaNs
                qt_intervals = (t_offsets - q_peaks) / desired_sampling_rate
                # Filter valid values
                valid_qt = ~np.isnan(qt_intervals) & (qt_intervals > 0)
                if np.any(valid_qt):
                    valid_qt_intervals = qt_intervals[valid_qt]
                    lead_features['Mean_QT_Interval_ms'] = np.nanmean(valid_qt_intervals) * 1000
                    lead_features['Std_QT_Interval_ms'] = np.nanstd(valid_qt_intervals) * 1000 if len(valid_qt_intervals) > 1 else 0
            
            # R-peak amplitude features
            if len(r_peaks) > 0:
                lead_features['Mean_R_Amplitude'] = np.nanmean(cleaned_signal[r_peaks])
                lead_features['Std_R_Amplitude'] = np.nanstd(cleaned_signal[r_peaks])
            
            # NEW: P Wave Amplitude Features
            if 'ECG_P_Peaks' in waves_info:
                p_peaks = np.array(waves_info['ECG_P_Peaks'])
                
                # Remove NaN values
                valid_p = ~np.isnan(p_peaks)
                if np.any(valid_p):
                    valid_p_indices = p_peaks[valid_p].astype(int)
                    
                    # Ensure indices are within bounds
                    valid_p_indices = valid_p_indices[(valid_p_indices >= 0) & (valid_p_indices < len(cleaned_signal))]
                    
                    if len(valid_p_indices) > 0:
                        # P wave amplitudes
                        p_amplitudes = cleaned_signal[valid_p_indices]
                        lead_features['Mean_P_Amplitude'] = np.nanmean(p_amplitudes)
                        lead_features['Std_P_Amplitude'] = np.nanstd(p_amplitudes) if len(p_amplitudes) > 1 else 0.0
            
            # NEW: P Wave Duration
            if 'ECG_P_Onsets' in waves_info and 'ECG_P_Offsets' in waves_info:
                p_onsets = np.array(waves_info['ECG_P_Onsets'])
                p_offsets = np.array(waves_info['ECG_P_Offsets'])
                
                # Remove NaN values
                valid_onsets = ~np.isnan(p_onsets)
                valid_offsets = ~np.isnan(p_offsets)
                
                # Calculate P durations only for valid onset/offset pairs
                if np.any(valid_onsets) and np.any(valid_offsets):
                    # Find min number of valid points
                    valid_on_indices = p_onsets[valid_onsets].astype(int)
                    valid_off_indices = p_offsets[valid_offsets].astype(int)
                    
                    # Ensure indices are within bounds
                    valid_on_indices = valid_on_indices[(valid_on_indices >= 0) & (valid_on_indices < len(cleaned_signal))]
                    valid_off_indices = valid_off_indices[(valid_off_indices >= 0) & (valid_off_indices < len(cleaned_signal))]
                    
                    # Match P onsets with their corresponding offsets
                    p_durations = []
                    for on_idx in valid_on_indices:
                        # Find the closest following offset
                        following_offs = valid_off_indices[valid_off_indices > on_idx]
                        if len(following_offs) > 0:
                            p_durations.append((following_offs[0] - on_idx) / desired_sampling_rate)
                    
                    if len(p_durations) > 0:
                        lead_features['Mean_P_Duration_ms'] = np.nanmean(p_durations) * 1000
                        lead_features['Std_P_Duration_ms'] = np.nanstd(p_durations) * 1000 if len(p_durations) > 1 else 0.0
            
            # QRS Net Deflection (Axis approximation)
            if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info and len(r_peaks) > 0:
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                
                # Remove NaN values
                valid_s = s_peaks[~np.isnan(s_peaks)]
                valid_q = q_peaks[~np.isnan(q_peaks)]
                
                if len(valid_s) > 0 and len(valid_q) > 0:
                    # Calculate mean QRS deflection (R-peak minus average of Q and S)
                    valid_s_indices = valid_s.astype(int)
                    valid_q_indices = valid_q.astype(int)
                    
                    # Ensure indices are within bounds
                    valid_s_indices = valid_s_indices[valid_s_indices < len(cleaned_signal)]
                    valid_q_indices = valid_q_indices[valid_q_indices < len(cleaned_signal)]
                    
                    if len(valid_s_indices) > 0 and len(valid_q_indices) > 0:
                        s_amplitudes = cleaned_signal[valid_s_indices]
                        q_amplitudes = cleaned_signal[valid_q_indices]
                        r_amplitudes = cleaned_signal[r_peaks.astype(int)]
                        
                        # QRS net deflection = R - average(Q+S)/2
                        mean_qs = (np.nanmean(q_amplitudes) + np.nanmean(s_amplitudes)) / 2
                        qrs_net_deflection = np.nanmean(r_amplitudes) - mean_qs
                        lead_features['QRS_Net_Deflection'] = qrs_net_deflection
                    else:
                        lead_features['QRS_Net_Deflection'] = np.nan
                else:
                    lead_features['QRS_Net_Deflection'] = np.nan
            else:
                lead_features['QRS_Net_Deflection'] = np.nan
            
        except Exception:
            # Silent fail to improve speed
            pass
        
    except Exception:
        # Silent fail to improve speed
        pass
        
    return lead_idx, lead_features

def process_leads_in_parallel(signals, n_leads, original_fs, desired_sampling_rate, expected_length, feature_names):
    """Process all leads in parallel using CPU cores."""
    # Prepare lead data for parallel processing
    lead_signals = [signals[:, i] for i in range(n_leads)]
    
    # Create a partially bound function with fixed parameters
    process_lead = partial(
        process_single_lead,
        original_fs=original_fs,
        desired_sampling_rate=desired_sampling_rate,
        expected_length=expected_length,
        feature_names=feature_names
    )
    
    # Determine number of cores to use - one per lead, but limited by system cores
    num_cores = min(n_leads, multiprocessing.cpu_count())
    
    # Process leads in parallel
    features_list = [{name: np.nan for name in feature_names} for _ in range(n_leads)]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Map each lead to a process with its index
        results = pool.starmap(
            process_lead,
            [(lead_signals[i], i) for i in range(n_leads)]
        )
    
    # Reorganize results in correct lead order
    for lead_idx, lead_features in results:
        features_list[lead_idx] = lead_features
        
    return features_list

def process_single_record(record, desired_sampling_rate=100, train=True, feature_medians=None):
    """Process a single ECG record and extract all features."""
    # Load signal data
    signals, fields = load_signals(record)
    # print(f"Processing record: {record}, signals shape: {signals.shape}")
    original_fs = fields["fs"]
    
    # Calculate expected length once
    expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
    # Setup - Updated with 10 features now (including P wave features)
    n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
    feature_names = [
        'Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms',
        'Mean_QT_Interval_ms', 'Std_QT_Interval_ms',
        'Mean_R_Amplitude', 'Std_R_Amplitude',
        'RR_Interval_RMSSD_ms', 'QRS_Net_Deflection',
        'Mean_P_Amplitude', 'Std_P_Amplitude',      # New P wave amplitude features
        'Mean_P_Duration_ms', 'Std_P_Duration_ms'   # New P wave duration features
    ]
    
    # Reshape if needed
    if len(signals.shape) == 1:
        signals = signals.reshape(-1, 1)
    
    # Process each lead in parallel
    features_list = process_leads_in_parallel(signals, n_leads, original_fs, 
                                           desired_sampling_rate, expected_length, feature_names)
    
    # Convert to numpy array
    features_array = np.array([list(d.values()) for d in features_list])
    
    # Imputation for missing features
    if train:
        # Calculate median for each feature column, ignoring NaNs
        medians_by_lead = []
        for lead_idx in range(features_array.shape[0]):
            lead_medians = {}
            for j, feature_name in enumerate(feature_names):
                lead_medians[feature_name] = np.nanmedian(features_array[lead_idx, j])
                # Handle case where all values are NaN
                if np.isnan(lead_medians[feature_name]):
                    lead_medians[feature_name] = 0.0
            medians_by_lead.append(lead_medians)
        
        # Impute missing values using vectorized operations
        for lead_idx in range(features_array.shape[0]):
            for j, feature_name in enumerate(feature_names):
                if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
                    features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
        
        feature_medians = medians_by_lead
        
    elif feature_medians is not None:
        # Apply saved medians to test data
        for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
            for j, feature_name in enumerate(feature_names):
                if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
                    features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
    
    # Final safety using vectorized operation
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    if train:
        label = load_label(record)
        label = np.array(label).astype(np.long)
        source = get_source(load_header(record))
        return record, features_array, label, source, feature_medians
    else:
        return record, features_array

def load_and_process_signal_train(records, 
                                  desired_sampling_rate=100,
                                  train=True,
                                  feature_medians=None,
                                  batch_size=5):

    # If records is a single record path, convert to list
    if isinstance(records, str):
        single_record = True
        records = [records]
    else:
        single_record = False
    
    results = []
    
    # Process records in batches
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: records {i+1} to {min(i+batch_size, len(records))}")
        
        # Use ProcessPoolExecutor to process records in parallel
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for record in batch_records:
                future = executor.submit(
                    process_single_record,
                    record=record,
                    desired_sampling_rate=desired_sampling_rate,
                    train=train,
                    feature_medians=feature_medians
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    record_result = future.result()
                    results.append(record_result)
                except Exception as e:
                    print(f"Error processing record: {e}")
    
    # If processing a single record, return only its results
    if single_record and results:
        # For training mode
        if train and len(results[0]) == 5:
            _, features_array, label, source, feature_medians = results[0]
            
            return features_array, label, source, feature_medians
        # For testing mode
        elif not train and len(results[0]) == 2:
            _, features_array = results[0]
            return features_array
    

    return results


# class CMLoss(nn.Module):
#     """
#     Clinical Metric (CM) Loss from Vicar et al. CinC2020-189.

#     The true challenge metric is:
#         CM = sum_{i,j} A_{ij} * W_{ij}
#     where A is the (soft) confusion matrix and W is the provided weight matrix.

#     This loss returns:
#         L = -CM
#     so that minimizing L maximizes the challenge metric.
#     """

#     def __init__(self, weight_matrix: torch.Tensor, eps: float = 1e-6):
#         """
#         Args:
#             weight_matrix: FloatTensor of shape (c, c) containing w_{ij}.
#             eps: Small constant to stabilize division.
#         """
#         super().__init__()
#         self.W = weight_matrix
#         self.eps = eps

#     def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             logits: FloatTensor of shape (batch, c), raw model outputs.
#             labels: FloatTensor of shape (batch, c), one-hot (or multi-hot) ground truth.

#         Returns:
#             loss: Scalar tensor = - sum_{i,j} A_{ij} * W_{ij}
#         """
#         # 1) Convert logits to probabilities
#         p = torch.sigmoid( logits )         # shape (b, c)
#         L = labels.float()                # shape (b, c)

#         # 2) Continuous OR: B = L OR p = L + p - L*p
#         B = L + p - L * p                 # shape (b, c)

#         # 3) Normalizer N: sum across classes then expand
#         N = B.sum(dim=1, keepdim=True)    # shape (b, 1)
#         N = N.clamp(min=self.eps).expand_as(B)  # shape (b, c)

#         # 4) Soft confusion matrix A = L^T @ (p / N)
#         R_over_N = p / N                  # shape (b, c)
#         A = L.t() @ R_over_N              # shape (c, c)

#         # 5) Weighted sum with W
#         cm_value = (A * self.W).sum()     # scalar

#         # 6) Return negative for minimization
#         return -cm_value

def train_model(data_folder, model_folder, verbose):
    """
    Train a model using the data in the data_folder and save it in the model_folder.
    Includes validation set with challenge score optimization.
    
    Args:
        data_folder: Folder containing the ECG records
        model_folder: Folder to save the trained model
        verbose: Whether to print progress
    """
    # Find the data files
    print("New submission: Training the model...")
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler()
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)
    print(f"Found {num_records} records")

    # Get full paths for records
    record_paths = [os.path.join(data_folder, record) for record in records]

    # Process records in batches of 5 (optimal for EC2 g4dn.4xlarge with 16 vCPUs)
    print("Processing records in parallel batches...")
    results = load_and_process_signal_train(
        record_paths,
        desired_sampling_rate=100,
        train=True,
        feature_medians=None,
        batch_size=5  # Process 5 records at a time
    )
    
    # Extract features, labels, and other data
    X_train_features = []
    y_train = []
    sources = []
    all_feature_medians = None

    for record_path, features, label, source, feature_medians in results:
        X_train_features.append(features)
        y_train.append(label)
        sources.append(source)
    
        # Initialize dictionary to collect all feature values for global median calculation
        if all_feature_medians is None:
            # Initialize storage for all values
            feature_collector = {}
            for lead_idx in range(len(feature_medians)):
                for feature_name in feature_medians[lead_idx].keys():
                    feature_collector[(lead_idx, feature_name)] = []
            all_feature_medians = feature_collector

        # Collect feature values from all records
        for lead_idx in range(len(feature_medians)):
            for feature_name, value in feature_medians[lead_idx].items():
                if not np.isnan(value) and not np.isinf(value):
                    all_feature_medians[(lead_idx, feature_name)].append(value)

    # Extract unique feature names (strings only) - MOVED OUTSIDE THE LOOP
    unique_feature_names = set()
    for key in all_feature_medians.keys():
        lead_idx, feature_name = key
        unique_feature_names.add(feature_name)

    # Calculate global medians - MOVED OUTSIDE THE LOOP
    global_medians = []
    for lead_idx in range(12):  # 12 leads
        lead_medians = {}
        for feature_name in unique_feature_names:
            values = all_feature_medians.get((lead_idx, feature_name), [])
            if values:
                lead_medians[feature_name] = np.median(values)
            else:
                lead_medians[feature_name] = 0.0  # Fallback default
        global_medians.append(lead_medians)

    all_feature_medians = global_medians
    # Convert lists to numpy arrays
    X_train_features = np.array(X_train_features)
    y_train = np.array(y_train)
    
    print(f"X_train_features shape: {X_train_features.shape}")  # Should be (num_samples, 12, 6)
    print(f"y_train shape: {y_train.shape}")

    # # # ################################################################################################
    # # # # Optionally save/load features to avoid reprocessing
    # np.savez("fold_1_training_features_new_p_median.npz", X_train_features=X_train_features, y_train=y_train, 
    #          all_feature_medians=all_feature_medians, sources=sources) 
    
    # data = np.load("fold_2_training_features_new_p.npz", allow_pickle=True)
    # X_train_features = data['X_train_features']
    # y_train = data['y_train']
    # all_feature_medians = data['all_feature_medians']
    # sources = data['sources'].tolist()
    ################################################################################################
    # from imblearn.over_sampling import SMOTE

    # smote = SMOTE(
    #     sampling_strategy='auto',  # Balance all classes to match majority class
    #     random_state=42,
    #     k_neighbors=5,  # Number of nearest neighbors to use
    # )


    # Split data into train and validation sets with specific class distribution
    # First, separate data by class
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]
    
    # Determine validation set size (20% of total data)
    val_size = int(0.05 * len(y_train))
    
    # Validation set: 95% class 0, 5% class 1
    val_class_0_size = int(0.95 * val_size)
    val_class_1_size = val_size - val_class_0_size
    
    # Ensure we don't request more samples than available
    val_class_0_size = min(val_class_0_size, len(class_0_indices))
    val_class_1_size = min(val_class_1_size, len(class_1_indices))
    
    # Randomly select samples for validation
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    val_class_0_indices = class_0_indices[:val_class_0_size]
    val_class_1_indices = class_1_indices[:val_class_1_size]
    
    # Combine validation indices and the remaining for training
    val_indices = np.concatenate([val_class_0_indices, val_class_1_indices])
    train_indices = np.setdiff1d(np.arange(len(y_train)), val_indices)
    
    # Create train and validation sets
    X_train = X_train_features[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train_features[val_indices]
    y_val = y_train[val_indices]
    
    print(f"Training set: {len(y_train_split)} samples ({np.sum(y_train_split == 0)} class 0, {np.sum(y_train_split == 1)} class 1)")
    print(f"Validation set: {len(y_val)} samples ({np.sum(y_val == 0)} class 0, {np.sum(y_val == 1)} class 1)")

    print(f"X_train shape: {X_train.shape}")  # Should be (num_samples, 12, 6)
    
    # Scale features - be careful to preserve the shape
    print("Starting scaling...")
    # Reshape for scaling, fitting on training data only
    original_train_shape = X_train.shape
    X_train_flat = X_train.reshape(original_train_shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler.transform(X_val_flat)
    
    # Reshape back to original dimensions
    X_train_scaled = X_train_scaled_flat.reshape(original_train_shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    
    # check if features are nan or not
    if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any():
        raise ValueError("NaN values found in scaled features. Check the input data and scaling process.")

    # Create tensor datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_split, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input dimension is number of features per lead (6)
    feature_per_lead = X_train.shape[2]  # Should be 6
    
    # Create model
    model = conv_model(input_dim=feature_per_lead, num_leads=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
                                       
    # Class weights calculation based on training split
    num_0s = np.sum(y_train_split == 0)
    num_1s = np.sum(y_train_split == 1)
    class_weights_0 = torch.tensor([1.0, num_0s/num_1s], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights_0}")
    
    focal_loss = FocalLoss(gamma=2, alpha=class_weights_0, reduction='mean', 
                           task_type='multi-class', num_classes=NUM_CLASSES).to(DEVICE)
    criterion = focal_loss.to(DEVICE)

    # # Create a 2x2 weight matrix that matches the challenge metric
    # Example: Challenge metric values for [[TN, FP], [FN, TP]]
    # weight_matrix = torch.tensor([
    #     [1.0, -0.5],  # Penalties for actual negative class
    #     [-2.0, 2.0]   # Penalties for actual positive class
    # ], dtype=torch.float32).to(DEVICE)

    # # Initialize CMLoss with this weight matrix
    # cm_loss_fn = CMLoss(weight_matrix=weight_matrix).to(DEVICE)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    # total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=EPOCHS*.1, num_training_steps=EPOCHS)

    if verbose:
        print('Starting training...')
    print("without penality")
    # Track best validation challenge score
    best_challenge_score = -1.0
    patience = 50  # Number of epochs to wait for improvement
    patience_counter = 0
    
    # Import the official challenge score function
    from helper_code import compute_challenge_score
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        all_train_predictions = []
        all_train_labels = []
        
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            f_loss = criterion(outputs, batch_labels)
            scores = F.softmax(outputs, dim=1)[:, 1] 

            # Compute ranking hinge loss
            ranking_loss = ranking_hinge_loss(scores, batch_labels, margin=2.0, num_pairs=10000)
            
            # #CMloss
            # Convert labels to one-hot encoding for CMLoss
            # one_hot_labels = F.one_hot(batch_labels, num_classes=NUM_CLASSES).float()
            # cm_loss = cm_loss_fn(outputs, one_hot_labels)  # Pass one-hot labels instead of class indices

            # Combined loss
            # print(f_loss, ranking_loss, cm_loss )
            loss =  f_loss + (1 * ranking_loss) # +.0001*cm_loss
            
            

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_labels.cpu().numpy())

        scheduler.step()  # Update learning rate after each epoch

        # Compute training metrics
        epoch_loss /= len(train_dataset)
        train_accuracy = (torch.tensor(all_train_predictions) == torch.tensor(all_train_labels)).sum().item() / len(all_train_labels)
        train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro')
        
        print(f'Training - Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}')
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_probabilities = []
        val_ground_truth = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_dataloader:
                batch_features = batch_features.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                outputs = model(batch_features)
                probabilities = F.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability for positive class
                val_ground_truth.extend(batch_labels.cpu().numpy())
        
        # Compute validation metrics
        val_accuracy = (torch.tensor(val_predictions) == torch.tensor(val_ground_truth)).sum().item() / len(val_ground_truth)
        val_f1 = f1_score(val_ground_truth, val_predictions, average='macro')
        
        # Calculate challenge score on validation set
        # Use fewer permutations during training for speed
        val_challenge_score = compute_challenge_score(
            val_ground_truth, 
            val_probabilities,
            num_permutations=10**4  # Reduced from 10^4 for faster computation during training
        )

        
        print(f'Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}, Challenge Score: {val_challenge_score:.4f}')
        
        # Save model if challenge score improves
        if val_challenge_score >= best_challenge_score:
            best_challenge_score = val_challenge_score
            patience_counter = 0
            print(f"New best challenge score: {best_challenge_score:.4f}, saving model...")
            
            # Save the model
            os.makedirs(model_folder, exist_ok=True)    
            save_model(model_folder, model, scaler, all_feature_medians)
        else:
            patience_counter += 1
            print(f"Challenge score did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        print(f"Best validation challenge score so far: {best_challenge_score:.4f}")
        if torch.isnan(loss):
            print(f"WARNING: Loss became NaN at epoch {epoch+1}")
            print("Last batch details:")
            print(f"  - Focal loss: {f_loss.item() if not torch.isnan(f_loss) else 'NaN'}")
            print(f"  - Ranking loss: {ranking_loss.item() if not torch.isnan(ranking_loss) else 'NaN'}")
            
            # If we have a good model already, just use it
            if best_challenge_score > 0.3:
                print(f"Using previously saved model with score: {best_challenge_score:.4f}")
                break
            else:
                # No good model yet, but training has become unstable
                print(f"No good model found yet (best: {best_challenge_score:.4f}). Training is unstable.")
                print("Restarting from the latest checkpoint with lower learning rate might help.")
                break  # It's usually better to break and restart with different settings
  
    if best_challenge_score < 0.3:
        print("Warning: Best validation challenge score is below 0.3, indicating poor model performance.")
        print(f"Best challenge score achieved: {best_challenge_score:.4f}")
        raise ValueError("Model training did not achieve a satisfactory challenge score.")
    
    print(f"Training complete. Best validation challenge score: {best_challenge_score:.4f}")



    # ########################################################################################
    # ########## 2nd cycle
    # del optimizer, scheduler, model

    # print('***********'*15)
    # print('starting cycle 2')

    # # Create a 2x2 weight matrix that matches the challenge metric
    # # Example: Challenge metric values for [[TN, FP], [FN, TP]]
    # weight_matrix = torch.tensor([
    #     [1.0, -2.0],  # Penalties for actual negative class
    #     [-1.0, 5.0]   # Penalties for actual positive class
    # ], dtype=torch.float32).to(DEVICE)

    # # weight_matrix = torch.tensor([
    # #     [1.0, -0.5],  # Penalties for actual negative class
    # #     [-1.0, 2.0]   # Penalties for actual positive class
    # # ], dtype=torch.float32).to(DEVICE)

    # # Initialize CMLoss with this weight matrix
    # cm_loss_fn = CMLoss(weight_matrix=weight_matrix).to(DEVICE)

    # model = conv_model(input_dim=12, num_leads=12, num_classes=NUM_CLASSES).to(DEVICE)  # Adjust input_dim and num_leads as needed

    # model_path = os.path.join(model_folder, 'model.pth')
    # if not os.path.exists(model_path):
    #     model_path = os.path.join(model_folder, 'transformer_model.pt')  # Try alternative name

    # model.load_state_dict(torch.load(model_path, map_location=DEVICE))


    # lr_c2 = 1e-4
    # optimizer = optim.AdamW(model.parameters(), lr=lr_c2, weight_decay=0.1)

    # # total_steps = len(train_dataloader) * EPOCHS
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 
    #                                         num_warmup_steps=EPOCHS*.1, num_training_steps=EPOCHS)
    


    # for epoch in range(EPOCHS):
    #     print(f'Epoch {epoch+1}/{EPOCHS}')
        
    #     # Training phase
    #     model.train()
    #     epoch_loss = 0.0
    #     all_train_predictions = []
    #     all_train_labels = []
        
    #     for batch_features, batch_labels in train_dataloader:
    #         batch_features = batch_features.to(DEVICE)
    #         batch_labels = batch_labels.to(DEVICE)
            
    #         optimizer.zero_grad()
    #         outputs = model(batch_features)
    #         f_loss = criterion(outputs, batch_labels)
    #         scores = F.softmax(outputs, dim=1)[:, 1] 

    #         # Compute ranking hinge loss
    #         ranking_loss = ranking_hinge_loss(scores, batch_labels, margin=2.0, num_pairs=10000)
            
    #         # #CMloss
    #         # # Convert labels to one-hot encoding for CMLoss
    #         one_hot_labels = F.one_hot(batch_labels, num_classes=NUM_CLASSES).float()
    #         cm_loss = cm_loss_fn(outputs, one_hot_labels)  # Pass one-hot labels instead of class indices

    #         # Combined loss
    #         loss = cm_loss
            
    #         loss.backward()
    #         optimizer.step()
            
    #         epoch_loss += loss.item() * batch_features.size(0)
    #         _, predicted = torch.max(outputs, 1)
    #         all_train_predictions.extend(predicted.cpu().numpy())
    #         all_train_labels.extend(batch_labels.cpu().numpy())

    #     scheduler.step()  # Update learning rate after each epoch

    #     # Compute training metrics
    #     epoch_loss /= len(train_dataset)
    #     train_accuracy = (torch.tensor(all_train_predictions) == torch.tensor(all_train_labels)).sum().item() / len(all_train_labels)
    #     train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro')
        
    #     print(f'Training - Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}')
        
    #     # Validation phase
    #     model.eval()
    #     val_predictions = []
    #     val_probabilities = []
    #     val_ground_truth = []
        
    #     with torch.no_grad():
    #         for batch_features, batch_labels in val_dataloader:
    #             batch_features = batch_features.to(DEVICE)
    #             batch_labels = batch_labels.to(DEVICE)
                
    #             outputs = model(batch_features)
    #             probabilities = F.softmax(outputs, dim=1)
                
    #             _, predicted = torch.max(outputs, 1)
                
    #             val_predictions.extend(predicted.cpu().numpy())
    #             val_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability for positive class
    #             val_ground_truth.extend(batch_labels.cpu().numpy())
        
    #     # Compute validation metrics
    #     val_accuracy = (torch.tensor(val_predictions) == torch.tensor(val_ground_truth)).sum().item() / len(val_ground_truth)
    #     val_f1 = f1_score(val_ground_truth, val_predictions, average='macro')
        
    #     # Calculate challenge score on validation set
    #     # Use fewer permutations during training for speed
    #     val_challenge_score = compute_challenge_score(
    #         val_ground_truth, 
    #         val_probabilities,
    #         num_permutations=10**4  # Reduced from 10^4 for faster computation during training
    #     )

        
    #     print(f'Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}, Challenge Score: {val_challenge_score:.4f}')
        
    #     # Save model if challenge score improves
    #     if val_challenge_score >= best_challenge_score:
    #         best_challenge_score = val_challenge_score
    #         patience_counter = 0
    #         print(f"New best challenge score: {best_challenge_score:.4f}, saving model...")
            
    #         # Save the model
    #         os.makedirs(model_folder, exist_ok=True)    
    #         save_model(model_folder, model, scaler, all_feature_medians)
    #     else:
    #         patience_counter += 1
    #         print(f"Challenge score did not improve. Patience: {patience_counter}/{patience}")
            
    #         if patience_counter >= patience:
    #             print(f"Early stopping after {patience} epochs without improvement")
    #             break
    #     print(f"Best validation challenge score so far: {best_challenge_score:.4f}")
    #     if torch.isnan(loss):
    #         print(f"WARNING: Loss became NaN at epoch {epoch+1}")
    #         print("Last batch details:")
    #         print(f"  - Focal loss: {f_loss.item() if not torch.isnan(f_loss) else 'NaN'}")
    #         print(f"  - Ranking loss: {ranking_loss.item() if not torch.isnan(ranking_loss) else 'NaN'}")
            
    #         # If we have a good model already, just use it
    #         if best_challenge_score > 0.3:
    #             print(f"Using previously saved model with score: {best_challenge_score:.4f}")
    #             break
    #         else:
    #             # No good model yet, but training has become unstable
    #             print(f"No good model found yet (best: {best_challenge_score:.4f}). Training is unstable.")
    #             print("Restarting from the latest checkpoint with lower learning rate might help.")
    #             break  # It's usually better to break and restart with different settings
  
    # if best_challenge_score < 0.3:
    #     print("Warning: Best validation challenge score is below 0.3, indicating poor model performance.")
    #     print(f"Best challenge score achieved: {best_challenge_score:.4f}")
    #     raise ValueError("Model training did not achieve a satisfactory challenge score.")
    
    # print(f"Training complete. Best validation challenge score: {best_challenge_score:.4f}")


            
    
    if verbose:
        print('Done.')
        print()

def run_model(record, model, verbose):
    """
    Run a trained model on a record.
    
    Args:
        record: Path to the record
        model: Trained model
        verbose: Whether to print progress
    
    Returns:
        (binary_output, probability_output): Prediction and probability
    """
    if verbose:
        print(f'Processing record {record} for prediction...')
    
    # Extract features from the record
    features_array = load_and_process_signal_train(
        record, 
        desired_sampling_rate=100,
        train=False,
        feature_medians=model.feature_medians  # Use the saved medians for imputation
    )
    
    # Apply scaling
    original_shape = features_array.shape
    features_flat = features_array.reshape(1, -1)  # Flatten for scaling
    features_scaled = model.feature_scaler.transform(features_flat)
    features_scaled = features_scaled.reshape(original_shape)  # Reshape back to (12, 6)
    
    # Convert to tensor and send to device
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # features_tensor shape: (1, 12, 6) - batch_size, num_leads, num_features
    
    # Run prediction
    with torch.no_grad():
        logits = model(features_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred_class = int(np.argmax(probs))
    
    binary_output = bool(pred_class)
    probability_output = float(probs[0, 1])  # Probability for the positive class
    
    if verbose:
        print(f'Prediction: {binary_output}, Probability: {probability_output:.4f}')
    
    return binary_output, probability_output

def load_model(model_folder, verbose):
    """
    Load a trained model from a folder.
    
    Args:
        model_folder: Folder containing the saved model
        verbose: Whether to print progress
    
    Returns:
        model: Loaded model
    """
    if verbose:
        print('Loading model and features preprocessing components...')
    
    # Load the PyTorch model
    model_path = os.path.join(model_folder, 'model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_folder, 'transformer_model.pt')  # Try alternative name
    
    # Load feature scaler
    feature_scaler_path = os.path.join(model_folder, 'feature_scaler.joblib')
    feature_scaler = joblib.load(feature_scaler_path)
    
    # Load feature medians for imputation
    feature_medians_path = os.path.join(model_folder, 'feature_medians.pkl')
    with open(feature_medians_path, 'rb') as f:
        feature_medians = pickle.load(f)
    
    # Create and load model
    model = conv_model(input_dim=12, num_leads=12, num_classes=NUM_CLASSES).to(DEVICE)  # Adjust input_dim and num_leads as needed
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    # Store preprocessing components with the model
    model.feature_scaler = feature_scaler
    model.feature_medians = feature_medians
    
    if verbose:
        print('Model loaded successfully.')
    return model

def save_model(model_folder, model, scaler, feature_medians):
    """
    Save model and related components to the model folder.
    
    Args:
        model_folder: Directory to save the model
        model: PyTorch model
        scaler: Feature scaler
        feature_medians: Median values for feature imputation
    """
    # Create directory if it doesn't exist
    os.makedirs(model_folder, exist_ok=True)
    
    # 1. Save the PyTorch model separately
    torch.save(model.state_dict(), os.path.join(model_folder, 'model.pth'))
    
    # 2. Save the scaler using joblib (better for scikit-learn objects)
    joblib.dump(scaler, os.path.join(model_folder, 'feature_scaler.joblib'))
    
    # 3. Save feature medians using pickle
    with open(os.path.join(model_folder, 'feature_medians.pkl'), 'wb') as f:
        pickle.dump(feature_medians, f)
    
    # 4. Also save a combined checkpoint for backward compatibility
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_medians': feature_medians
    }, os.path.join(model_folder, 'transformer_model.pt'))
    
    print(f"Model and components saved to {model_folder}")