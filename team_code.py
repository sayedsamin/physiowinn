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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
LEARNING_RATE = 0.0001 # it was 0.001 for transformer
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


# class ECGFeatureTransformer(nn.Module):
#     def __init__(self, feature_dim=6, num_leads=12, n_heads=8, n_layers=2, dropout=0.1, num_classes=2):
#         super(ECGFeatureTransformer, self).__init__()
        
#         self.feature_dim = feature_dim
#         self.num_leads = num_leads
        
#         # Feature embedding - increase dimensionality for transformer
#         self.feature_embedding = nn.Linear(feature_dim, 64)
        
#         # Positional encoding for leads
#         self.pos_encoding = nn.Parameter(torch.zeros(1, num_leads, 64))
#         nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
#         # Transformer operates on leads as sequence length
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=64,  # Embedding dimension
#                 nhead=n_heads,
#                 dim_feedforward=256,
#                 dropout=dropout,
#                 batch_first=True
#             ),
#             num_layers=n_layers
#         )
        
#         # Global attention pooling
#         self.attention = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.Tanh(),
#             nn.Linear(32, 1)
#         )
        
#         # Final classification layers
#         self.classifier = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, num_classes)
#         )
    
#     def forward(self, x):
#         # x shape: (batch_size, num_leads, feature_dim)
#         batch_size = x.size(0)
        
#         # Project each lead's features to higher dimension
#         x = self.feature_embedding(x)  # (batch_size, num_leads, 64)
        
#         # Add positional encoding for leads
#         x = x + self.pos_encoding
        
#         # Pass through transformer
#         x = self.transformer_encoder(x)  # (batch_size, num_leads, 64)
        
#         # Attention-based pooling over leads
#         attn_weights = self.attention(x)  # (batch_size, num_leads, 1)
#         attn_weights = F.softmax(attn_weights, dim=1)
#         x = torch.sum(x * attn_weights, dim=1)  # (batch_size, 64)
        
#         # Classification
#         output = self.classifier(x)
        
#         return output


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
            nn.ReLU(),
     
        )
        self.conv2 = nn.Sequential(
            self.conv2_layer,
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1))  # Downsample
        )

        # Fully connected layers
        self.fc1 = nn.Linear(1536, 128)  # Adjust input size based on conv output
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, num_leads, input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)  # Flatten the output for fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



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

# def load_and_process_signal_train(record,
#                                   desired_sampling_rate=100,
#                                   train=True,
#                                   feature_medians=None):
#     """
#     Load, preprocess ECG signals and extract features using NeuroKit2.
    
#     Args:
#         record: The record identifier/path
#         desired_sampling_rate: Target sampling rate (default: 100 Hz)
#         train: Whether this is training data
#         feature_medians: Dictionary of median values for imputation (for test data)
    
#     Returns:
#         If train=True: features_array, label, source, feature_medians
#         If train=False: features_array
#     """
#     signals, fields = load_signals(record)
#     original_fs = fields["fs"]
    
#     # Calculate expected length after resampling once (for consistency)
#     expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
#     # 1) Process all leads
#     n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
#     processed_signals = np.zeros((expected_length, n_leads))
#     features_list = []
#     feature_names = [
#         'Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms',
#         'Mean_QT_Interval_ms', 'Std_QT_Interval_ms',
#         'Mean_R_Amplitude', 'Std_R_Amplitude'
#     ]
    
#     # If signals is single-lead, reshape for consistency
#     if len(signals.shape) == 1:
#         signals = signals.reshape(-1, 1)
    
#     # Process each lead with NeuroKit2
#     print(f"\nProcessing {n_leads} leads with NeuroKit2...")
#     for i in range(n_leads):
#         ecg_signal = signals[:, i]
#         lead_features = {name: np.nan for name in feature_names}
        
#         try:
#             # 1. Normalize Signal to [-1, 1]
#             normalized_signal = normalize_signal(ecg_signal)
            
#             # 2. Resample using NeuroKit2 with EXACT length specified
#             resampled_signal = nk.signal_resample(
#                 normalized_signal,
#                 sampling_rate=original_fs,
#                 desired_sampling_rate=desired_sampling_rate,
#                 desired_length=expected_length  # Force exact length
#             )
            
#             # Verify length is as expected
#             if len(resampled_signal) != expected_length:
#                 print(f"Warning: Resampled signal length {len(resampled_signal)} doesn't match expected {expected_length}")
#                 # Force correct length
#                 if len(resampled_signal) < expected_length:
#                     # Pad with edge values
#                     resampled_signal = np.pad(resampled_signal, (0, expected_length - len(resampled_signal)), 'edge')
#                 else:
#                     # Truncate
#                     resampled_signal = resampled_signal[:expected_length]
            
#             # 3. Clean using NeuroKit2 (applies appropriate filtering)
#             cleaned_signal = nk.ecg_clean(resampled_signal, 
#                                           sampling_rate=desired_sampling_rate, 
#                                           method="neurokit")
            
#             # Store the processed signal
#             processed_signals[:, i] = cleaned_signal
            
#             # 4. Find R-Peaks
#             _, info = nk.ecg_peaks(cleaned_signal, sampling_rate=desired_sampling_rate)
#             r_peaks = info['ECG_R_Peaks']
            
#             if len(r_peaks) < 2:
#                 print(f"Lead {i+1}: Not enough R-peaks found. Will use median imputation.")
#                 features_list.append(lead_features)  # Add NaN features for now
#                 continue
                
#             # 5. Delineate Waveforms
#             try:
#                 _, waves_info = nk.ecg_delineate(cleaned_signal, r_peaks, 
#                                                 sampling_rate=desired_sampling_rate, method="dwt")
                
#                 # 6. Calculate Features with safety checks
#                 # Check if required wave points exist
#                 if ('ECG_S_Peaks' not in waves_info or 'ECG_Q_Peaks' not in waves_info or
#                     'ECG_T_Offsets' not in waves_info):
#                     print(f"Lead {i+1}: Missing required wave points. Using median imputation.")
#                     features_list.append(lead_features)
#                     continue
                
#                 # Convert lists to arrays with safety
#                 s_peaks = np.array(waves_info['ECG_S_Peaks'])
#                 q_peaks = np.array(waves_info['ECG_Q_Peaks'])
#                 t_offsets = np.array(waves_info['ECG_T_Offsets'])
                
#                 # Remove NaNs or invalid indices
#                 valid_qrs = ~np.isnan(s_peaks) & ~np.isnan(q_peaks) & (s_peaks >= q_peaks)
#                 valid_qt = ~np.isnan(t_offsets) & ~np.isnan(q_peaks) & (t_offsets >= q_peaks)
                
#                 # Calculate intervals only for valid points
#                 if np.any(valid_qrs):
#                     qrs_durations = (s_peaks[valid_qrs] - q_peaks[valid_qrs]) / desired_sampling_rate
#                     lead_features['Mean_QRS_Duration_ms'] = np.mean(qrs_durations) * 1000
#                     lead_features['Std_QRS_Duration_ms'] = np.std(qrs_durations) * 1000 if len(qrs_durations) > 1 else 0
                
#                 if np.any(valid_qt):
#                     qt_intervals = (t_offsets[valid_qt] - q_peaks[valid_qt]) / desired_sampling_rate
#                     lead_features['Mean_QT_Interval_ms'] = np.mean(qt_intervals) * 1000
#                     lead_features['Std_QT_Interval_ms'] = np.std(qt_intervals) * 1000 if len(qt_intervals) > 1 else 0
                
#                 # R-peak amplitudes
#                 lead_features['Mean_R_Amplitude'] = np.mean(cleaned_signal[r_peaks])
#                 lead_features['Std_R_Amplitude'] = np.std(cleaned_signal[r_peaks])
                
#                 # print(f"Lead {i+1}: Processed and features extracted successfully.")
                
#             except Exception as e:
#                 print(f"Lead {i+1}: Error during delineation: {e}")
#                 # Keep NaN values for this lead
            
#         except Exception as e:
#             print(f"Lead {i+1}: Error during processing: {e}")
            
#         features_list.append(lead_features)
    
#     # Convert to numpy array (still with NaN values)
#     features_array = np.array([list(d.values()) for d in features_list])
    
#     # Safety check for NaN/Inf values
#     nan_count = np.isnan(features_array).sum()
#     inf_count = np.isinf(features_array).sum()
#     if nan_count > 0 or inf_count > 0:
#         print(f"Found {nan_count} NaN values and {inf_count} Inf values in features. Imputing...")
    
#     # TRAINING: Calculate and save median values for imputation
#     if train:
#         # Calculate median for each feature column, ignoring NaNs
#         medians_by_lead = []
#         for lead_idx in range(features_array.shape[0]):
#             lead_medians = {}
#             for j, feature_name in enumerate(feature_names):
#                 lead_medians[feature_name] = np.nanmedian(features_array[lead_idx, j])
#                 # Handle case where all values are NaN
#                 if np.isnan(lead_medians[feature_name]):
#                     lead_medians[feature_name] = 0.0
#             medians_by_lead.append(lead_medians)
        
#         # Impute missing values in training data
#         for lead_idx in range(features_array.shape[0]):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
        
#         # Save medians for test data
#         feature_medians = medians_by_lead
        
#     # TESTING: Apply saved median values
#     elif feature_medians is not None:
#         # Impute missing values in test data using training medians
#         for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
    
#     # Final safety: Replace any remaining NaN/Inf with zeros
#     features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
#     if train:
#         label = load_label(record)
#         label = np.array(label).astype(np.long)
#         source = get_source(load_header(record))
#         return features_array, label, source, feature_medians
#     else:
#         return features_array

# def load_and_process_signal_train(record,
#                                   desired_sampling_rate=100,
#                                   train=True,
#                                   feature_medians=None):
#     """
#     Optimized function to load, preprocess ECG signals and extract features.
#     Uses nanmean/nanstd to handle missing values more efficiently.
    
#     Args:
#         record: The record identifier/path
#         desired_sampling_rate: Target sampling rate (default: 100 Hz)
#         train: Whether this is training data
#         feature_medians: Dictionary of median values for imputation (for test data)
    
#     Returns:
#         If train=True: features_array, label, source, feature_medians
#         If train=False: features_array
#     """
#     # Load signal data
#     signals, fields = load_signals(record)
#     original_fs = fields["fs"]
    
#     # Calculate expected length once
#     expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
#     # Setup
#     n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
#     features_list = []
#     feature_names = [
#         'Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms',
#         'Mean_QT_Interval_ms', 'Std_QT_Interval_ms',
#         'Mean_R_Amplitude', 'Std_R_Amplitude'
#     ]
    
#     # Reshape if needed
#     if len(signals.shape) == 1:
#         signals = signals.reshape(-1, 1)
    
#     # Reduce print statements for speed
#     # print(f"\nProcessing {n_leads} leads...")
    
#     # Process each lead
#     for i in range(n_leads):
#         ecg_signal = signals[:, i]
#         lead_features = {name: np.nan for name in feature_names}
        
#         try:
#             # 1. Fast normalization
#             normalized_signal = normalize_signal(ecg_signal)
            
#             # 2. Faster resampling with exact length
#             resampled_signal = nk.signal_resample(
#                 normalized_signal,
#                 sampling_rate=original_fs,
#                 desired_sampling_rate=desired_sampling_rate,
#                 desired_length=expected_length
#             )
            
#             # 3. Use faster cleaning method - pantompkins is faster than neurokit
#             cleaned_signal = nk.ecg_clean(
#                 resampled_signal, 
#                 sampling_rate=desired_sampling_rate, 
#                 method="pantompkins"  # Faster method
#             )
            
#             # 4. Faster R-peak detection
#             _, info = nk.ecg_peaks(
#                 cleaned_signal, 
#                 sampling_rate=desired_sampling_rate,
#                 method="pantompkins"  # Faster method
#             )
#             r_peaks = info['ECG_R_Peaks']
            
#             if len(r_peaks) < 2:
#                 # Skip delineation if insufficient R-peaks
#                 features_list.append(lead_features)
#                 continue
                
#             # 5. Delineate waveforms
#             try:
#                 _, waves_info = nk.ecg_delineate(
#                     cleaned_signal, 
#                     r_peaks, 
#                     sampling_rate=desired_sampling_rate, 
#                     method="dwt"
#                 )
                
#                 # 6. Direct calculation with nanmean/nanstd
#                 if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info:
#                     s_peaks = np.array(waves_info['ECG_S_Peaks'])
#                     q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                    
#                     # Calculate QRS durations directly, handling NaNs
#                     qrs_durations = (s_peaks - q_peaks) / desired_sampling_rate
#                     # Filter valid values
#                     valid_qrs = ~np.isnan(qrs_durations) & (qrs_durations > 0)
#                     if np.any(valid_qrs):
#                         valid_qrs_durations = qrs_durations[valid_qrs]
#                         lead_features['Mean_QRS_Duration_ms'] = np.nanmean(valid_qrs_durations) * 1000
#                         lead_features['Std_QRS_Duration_ms'] = np.nanstd(valid_qrs_durations) * 1000 if len(valid_qrs_durations) > 1 else 0
                
#                 if 'ECG_T_Offsets' in waves_info and 'ECG_Q_Peaks' in waves_info:
#                     t_offsets = np.array(waves_info['ECG_T_Offsets'])
#                     q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                    
#                     # Calculate QT intervals directly, handling NaNs
#                     qt_intervals = (t_offsets - q_peaks) / desired_sampling_rate
#                     # Filter valid values
#                     valid_qt = ~np.isnan(qt_intervals) & (qt_intervals > 0)
#                     if np.any(valid_qt):
#                         valid_qt_intervals = qt_intervals[valid_qt]
#                         lead_features['Mean_QT_Interval_ms'] = np.nanmean(valid_qt_intervals) * 1000
#                         lead_features['Std_QT_Interval_ms'] = np.nanstd(valid_qt_intervals) * 1000 if len(valid_qt_intervals) > 1 else 0
                
#                 # R-peak amplitudes using nanmean/nanstd
#                 lead_features['Mean_R_Amplitude'] = np.nanmean(cleaned_signal[r_peaks])
#                 lead_features['Std_R_Amplitude'] = np.nanstd(cleaned_signal[r_peaks])
                
#             except Exception:
#                 # Silent fail to improve speed
#                 pass
            
#         except Exception:
#             # Silent fail to improve speed
#             pass
            
#         features_list.append(lead_features)
    
#     # Convert to numpy array
#     features_array = np.array([list(d.values()) for d in features_list])
    
#     # Still need some imputation for missing features
#     if train:
#         # Calculate median for each feature column, ignoring NaNs
#         medians_by_lead = []
#         for lead_idx in range(features_array.shape[0]):
#             lead_medians = {}
#             for j, feature_name in enumerate(feature_names):
#                 lead_medians[feature_name] = np.nanmedian(features_array[lead_idx, j])
#                 # Handle case where all values are NaN
#                 if np.isnan(lead_medians[feature_name]):
#                     lead_medians[feature_name] = 0.0
#             medians_by_lead.append(lead_medians)
        
#         # Impute missing values using vectorized operations
#         for lead_idx in range(features_array.shape[0]):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
        
#         feature_medians = medians_by_lead
        
#     elif feature_medians is not None:
#         # Apply saved medians to test data
#         for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
    
#     # Final safety using vectorized operation
#     features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
#     if train:
#         label = load_label(record)
#         label = np.array(label).astype(np.long)
#         source = get_source(load_header(record))
#         return features_array, label, source, feature_medians
#     else:
#         return features_array

# def load_and_process_signal_train(record,
#                                   desired_sampling_rate=100,
#                                   train=True,
#                                   feature_medians=None):
#     """
#     Optimized function to load and extract only QRS duration features.
    
#     Args:
#         record: The record identifier/path
#         desired_sampling_rate: Target sampling rate (default: 100 Hz)
#         train: Whether this is training data
#         feature_medians: Dictionary of median values for imputation (for test data)
    
#     Returns:
#         If train=True: features_array, label, source, feature_medians
#         If train=False: features_array
#     """
#     # Load signal data
#     signals, fields = load_signals(record)
#     original_fs = fields["fs"]
    
#     # Calculate expected length once
#     expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
#     # Setup
#     n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
#     features_list = []
    
#     # Simplified feature names - only QRS duration features
#     feature_names = [
#         'Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms'
#     ]
    
#     # Reshape if needed
#     if len(signals.shape) == 1:
#         signals = signals.reshape(-1, 1)
    
#     # Process each lead
#     for i in range(n_leads):
#         ecg_signal = signals[:, i]
#         lead_features = {name: np.nan for name in feature_names}
        
#         try:
#             # 1. Fast normalization
#             normalized_signal = normalize_signal(ecg_signal)
            
#             # 2. Faster resampling with exact length
#             resampled_signal = nk.signal_resample(
#                 normalized_signal,
#                 sampling_rate=original_fs,
#                 desired_sampling_rate=desired_sampling_rate,
#                 desired_length=expected_length
#             )
            
#             # 3. Use faster cleaning method - pantompkins is faster than neurokit
#             cleaned_signal = nk.ecg_clean(
#                 resampled_signal, 
#                 sampling_rate=desired_sampling_rate, 
#                 method="pantompkins"  # Faster method
#             )
            
#             # 4. Faster R-peak detection
#             _, info = nk.ecg_peaks(
#                 cleaned_signal, 
#                 sampling_rate=desired_sampling_rate,
#                 method="pantompkins"  # Faster method
#             )
#             r_peaks = info['ECG_R_Peaks']
            
#             if len(r_peaks) < 2:
#                 # Skip delineation if insufficient R-peaks
#                 features_list.append(lead_features)
#                 continue
                
#             # 5. Delineate waveforms
#             try:
#                 _, waves_info = nk.ecg_delineate(
#                     cleaned_signal, 
#                     r_peaks, 
#                     sampling_rate=desired_sampling_rate, 
#                     method="dwt"
#                 )
                
#                 # 6. Calculate only QRS durations
#                 if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info:
#                     s_peaks = np.array(waves_info['ECG_S_Peaks'])
#                     q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                    
#                     # Calculate QRS durations directly, handling NaNs
#                     qrs_durations = (s_peaks - q_peaks) / desired_sampling_rate
#                     # Filter valid values
#                     valid_qrs = ~np.isnan(qrs_durations) & (qrs_durations > 0)
#                     if np.any(valid_qrs):
#                         valid_qrs_durations = qrs_durations[valid_qrs]
#                         lead_features['Mean_QRS_Duration_ms'] = np.nanmean(valid_qrs_durations) * 1000
#                         lead_features['Std_QRS_Duration_ms'] = np.nanstd(valid_qrs_durations) * 1000 if len(valid_qrs_durations) > 1 else 0
                
#                 # Removed QT interval and R amplitude calculations
                
#             except Exception:
#                 # Silent fail to improve speed
#                 pass
            
#         except Exception:
#             # Silent fail to improve speed
#             pass
            
#         features_list.append(lead_features)
    
#     # Convert to numpy array
#     features_array = np.array([list(d.values()) for d in features_list])
    
#     # Still need some imputation for missing features
#     if train:
#         # Calculate median for each feature column, ignoring NaNs
#         medians_by_lead = []
#         for lead_idx in range(features_array.shape[0]):
#             lead_medians = {}
#             for j, feature_name in enumerate(feature_names):
#                 lead_medians[feature_name] = np.nanmedian(features_array[lead_idx, j])
#                 # Handle case where all values are NaN
#                 if np.isnan(lead_medians[feature_name]):
#                     lead_medians[feature_name] = 0.0
#             medians_by_lead.append(lead_medians)
        
#         # Impute missing values using vectorized operations
#         for lead_idx in range(features_array.shape[0]):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
        
#         feature_medians = medians_by_lead
        
#     elif feature_medians is not None:
#         # Apply saved medians to test data
#         for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
    
#     # Final safety using vectorized operation
#     features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
#     if train:
#         label = load_label(record)
#         label = np.array(label).astype(np.long)
#         source = get_source(load_header(record))
#         return features_array, label, source, feature_medians
#     else:
#         return features_array


# import multiprocessing
# from functools import partial

# def process_single_lead(lead_data, lead_idx, original_fs, desired_sampling_rate, expected_length):
#     """Process a single lead and extract QRS duration features."""
#     ecg_signal = lead_data
#     feature_names = ['Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms']
#     lead_features = {name: np.nan for name in feature_names}
    
#     try:
#         # 1. Fast normalization
#         normalized_signal = normalize_signal(ecg_signal)
        
#         # 2. Faster resampling with exact length
#         resampled_signal = nk.signal_resample(
#             normalized_signal,
#             sampling_rate=original_fs,
#             desired_sampling_rate=desired_sampling_rate,
#             desired_length=expected_length
#         )
        
#         # 3. Use faster cleaning method - pantompkins is faster than neurokit
#         cleaned_signal = nk.ecg_clean(
#             resampled_signal, 
#             sampling_rate=desired_sampling_rate, 
#             method="pantompkins"
#         )
        
#         # 4. Faster R-peak detection
#         _, info = nk.ecg_peaks(
#             cleaned_signal, 
#             sampling_rate=desired_sampling_rate,
#             method="pantompkins"
#         )
#         r_peaks = info['ECG_R_Peaks']
        
#         if len(r_peaks) < 2:
#             return lead_idx, lead_features
            
#         # 5. Delineate waveforms
#         try:
#             _, waves_info = nk.ecg_delineate(
#                 cleaned_signal, 
#                 r_peaks, 
#                 sampling_rate=desired_sampling_rate, 
#                 method="dwt"
#             )
            
#             # 6. Calculate only QRS durations
#             if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info:
#                 s_peaks = np.array(waves_info['ECG_S_Peaks'])
#                 q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                
#                 # Calculate QRS durations directly, handling NaNs
#                 qrs_durations = (s_peaks - q_peaks) / desired_sampling_rate
#                 # Filter valid values
#                 valid_qrs = ~np.isnan(qrs_durations) & (qrs_durations > 0)
#                 if np.any(valid_qrs):
#                     valid_qrs_durations = qrs_durations[valid_qrs]
#                     lead_features['Mean_QRS_Duration_ms'] = np.nanmean(valid_qrs_durations) * 1000
#                     lead_features['Std_QRS_Duration_ms'] = np.nanstd(valid_qrs_durations) * 1000 if len(valid_qrs_durations) > 1 else 0
            
#         except Exception:
#             # Silent fail to improve speed
#             pass
        
#     except Exception:
#         # Silent fail to improve speed
#         pass
        
#     return lead_idx, lead_features

# def load_and_process_signal_train(record,
#                                   desired_sampling_rate=100,
#                                   train=True,
#                                   feature_medians=None):
#     """
#     Parallelized function to load and extract QRS duration features.
#     Uses one CPU core per lead for faster processing.
    
#     Args:
#         record: The record identifier/path
#         desired_sampling_rate: Target sampling rate (default: 100 Hz)
#         train: Whether this is training data
#         feature_medians: Dictionary of median values for imputation (for test data)
    
#     Returns:
#         If train=True: features_array, label, source, feature_medians
#         If train=False: features_array
#     """
#     # Load signal data
#     signals, fields = load_signals(record)
#     original_fs = fields["fs"]
    
#     # Calculate expected length once
#     expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
#     # Setup
#     n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
#     feature_names = ['Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms']
    
#     # Reshape if needed
#     if len(signals.shape) == 1:
#         signals = signals.reshape(-1, 1)
    
#     # Prepare lead data for parallel processing
#     lead_signals = [signals[:, i] for i in range(n_leads)]
    
#     # Create a partially bound function with fixed parameters
#     process_lead = partial(
#         process_single_lead,
#         original_fs=original_fs,
#         desired_sampling_rate=desired_sampling_rate,
#         expected_length=expected_length
#     )
    
#     # Determine number of cores to use - one per lead, but limited by system cores
#     num_cores = min(n_leads, multiprocessing.cpu_count())
    
#     # Process leads in parallel
#     features_list = [{name: np.nan for name in feature_names} for _ in range(n_leads)]
    
#     with multiprocessing.Pool(processes=num_cores) as pool:
#         # Map each lead to a process with its index
#         results = pool.starmap(
#             process_lead,
#             [(lead_signals[i], i) for i in range(n_leads)]
#         )
    
#     # Reorganize results in correct lead order
#     for lead_idx, lead_features in results:
#         features_list[lead_idx] = lead_features
    
#     # Convert to numpy array
#     features_array = np.array([list(d.values()) for d in features_list])
    
#     # Imputation for missing features
#     if train:
#         # Calculate median for each feature column, ignoring NaNs
#         medians_by_lead = []
#         for lead_idx in range(features_array.shape[0]):
#             lead_medians = {}
#             for j, feature_name in enumerate(feature_names):
#                 lead_medians[feature_name] = np.nanmedian(features_array[lead_idx, j])
#                 # Handle case where all values are NaN
#                 if np.isnan(lead_medians[feature_name]):
#                     lead_medians[feature_name] = 0.0
#             medians_by_lead.append(lead_medians)
        
#         # Impute missing values using vectorized operations
#         for lead_idx in range(features_array.shape[0]):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
        
#         feature_medians = medians_by_lead
        
#     elif feature_medians is not None:
#         # Apply saved medians to test data
#         for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
#             for j, feature_name in enumerate(feature_names):
#                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
#                     features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
    
#     # Final safety using vectorized operation
#     features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
#     if train:
#         label = load_label(record)
#         label = np.array(label).astype(np.long)
#         source = get_source(load_header(record))
#         return features_array, label, source, feature_medians
#     else:
#         return features_array

import multiprocessing
from functools import partial
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import neurokit2 as nk
from helper_code import load_signals, load_label, load_header, get_source

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
    """Process a single lead and extract all requested features."""
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
            desired_length=expected_length
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
    original_fs = fields["fs"]
    
    # Calculate expected length once
    expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
    # Setup
    n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
    feature_names = [
        'Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms',
        'Mean_QT_Interval_ms', 'Std_QT_Interval_ms',
        'Mean_R_Amplitude', 'Std_R_Amplitude'
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
    """
    Process multiple ECG records in parallel batches.
    
    Args:
        records: List of record paths or a single record path
        desired_sampling_rate: Target sampling rate (default: 100 Hz)
        train: Whether this is training data
        feature_medians: Dictionary of median values for imputation
        batch_size: Number of records to process in parallel (default: 5)
    
    Returns:
        If single record and train=True: features_array, label, source, feature_medians
        If single record and train=False: features_array
        If multiple records: list of processed record results
    """
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

def train_model(data_folder, model_folder, verbose):
    """
    Train a model using the data in the data_folder and save it in the model_folder.
    
    Args:
        data_folder: Folder containing the ECG records
        model_folder: Folder to save the trained model
        verbose: Whether to print progress
    """
    # Find the data files.
    print("New submission: Training the model...")
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
      
         # Save the first record's feature medians or update
        if all_feature_medians is None:
            all_feature_medians = feature_medians
    
    # Convert lists to numpy arrays
    X_train_features = np.array(X_train_features)
    y_train = np.array(y_train)
    
    print(f"X_train_features shape: {X_train_features.shape}")  # Should be (num_samples, 12, 6)
    print(f"y_train shape: {y_train.shape}")

    ################################################################################################
    # save the features
    # np.savez("fold_2_training_features.npz", X_train_features=X_train_features, y_train=y_train, 
    #          all_feature_medians=all_feature_medians, sources = sources) 
    
    # data = np.load("fold_2_training_features.npz", allow_pickle=True)
    # X_train_features = data['X_train_features']
    # y_train = data['y_train']
    # all_feature_medians = data['all_feature_medians']  # Convert to dictionary
    # sources = data['sources'].tolist()  # Convert to list
    ####################################################################################################

    # Scale features - be careful to preserve the shape
    print("Starting scaling...")
    # Reshape for scaling: (samples, leads*features) -> scale -> (samples, leads, features)
    original_shape = X_train_features.shape
    X_train_features_flat = X_train_features.reshape(original_shape[0], -1)
    X_train_features_scaled = scaler.fit_transform(X_train_features_flat)
    X_train_features_scaled = X_train_features_scaled.reshape(original_shape)
    
    # Create tensor dataset
    dataset = TensorDataset(
        torch.tensor(X_train_features_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input dimension is number of features per lead (6)
    feature_per_lead = X_train_features.shape[2]  # Should be 6
    
    # Create model
    model = conv_model(input_dim=feature_per_lead, num_leads=X_train_features.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
                                       
    # Class weights calculation
    num_0s = np.sum(y_train == 0)
    num_1s = np.sum(y_train == 1)
    class_weights_0 = torch.tensor([1.0, num_0s/num_1s], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights_0}")
    
    focal_loss = FocalLoss(gamma=2, alpha=class_weights_0, reduction='mean', 
                           task_type='multi-class', num_classes=NUM_CLASSES).to(DEVICE)
    criterion = focal_loss.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if verbose:
        print('Starting training...')
    # print("using entropy loss")
    model.train()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            # #######
            # p = nn.Softmax(dim=1)(outputs)
            # entropy_per_ex = torch.sum(-p * torch.log(p + 1e-12), dim=1)
            # mask_pos = (batch_labels == 1)
            # if mask_pos.any():
            #     # Average entropy over only positive samples
            #     entropy_pos = entropy_per_ex[mask_pos].mean()
            # else:
            #     # No positive examples in this batch → no confidence penalty
            #     entropy_pos = torch.tensor(0., device=outputs.device)
            # #######
            # loss = loss + 0.1 * entropy_pos
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        # Compute metrics
        epoch_loss /= len(dataset)
        correct = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()
        accuracy = correct / len(all_labels)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        

        # save the model every 5 epochs and check for loss = nan, if loss is nan then dont save the model and use the previous model
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        # if np.isnan(epoch_loss):
        #     print(f"Epoch {epoch+1}: Loss is NaN, skipping model save.")
        #     continue
        if accuracy > 0.95:
            print(f"Epoch {epoch+1}: Accuracy is above 0.95, saving model.")
            # Save the model
            break        

        
        
    # Save model, scaler, and feature medians
    os.makedirs(model_folder, exist_ok=True)    
    save_model(model_folder, model, scaler, all_feature_medians)

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
    model = conv_model(input_dim=6, num_leads=12, num_classes=NUM_CLASSES).to(DEVICE)  # Adjust input_dim and num_leads as needed
    
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