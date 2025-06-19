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

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LEARNING_RATE = 0.001
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


class ECGFeatureTransformer(nn.Module):
    def __init__(self, feature_dim=6, num_leads=12, n_heads=8, n_layers=2, dropout=0.1, num_classes=2):
        super(ECGFeatureTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_leads = num_leads
        
        # Feature embedding - increase dimensionality for transformer
        self.feature_embedding = nn.Linear(feature_dim, 64)
        
        # Positional encoding for leads
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_leads, 64))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Transformer operates on leads as sequence length
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,  # Embedding dimension
                nhead=n_heads,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_leads, feature_dim)
        batch_size = x.size(0)
        
        # Project each lead's features to higher dimension
        x = self.feature_embedding(x)  # (batch_size, num_leads, 64)
        
        # Add positional encoding for leads
        x = x + self.pos_encoding
        
        # Pass through transformer
        x = self.transformer_encoder(x)  # (batch_size, num_leads, 64)
        
        # Attention-based pooling over leads
        attn_weights = self.attention(x)  # (batch_size, num_leads, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)  # (batch_size, 64)
        
        # Classification
        output = self.classifier(x)
        
        return output




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

def load_and_process_signal_train(record,
                                  desired_sampling_rate=100,
                                  train=True,
                                  feature_medians=None):
    """
    Load, preprocess ECG signals and extract features using NeuroKit2.
    
    Args:
        record: The record identifier/path
        desired_sampling_rate: Target sampling rate (default: 100 Hz)
        train: Whether this is training data
        feature_medians: Dictionary of median values for imputation (for test data)
    
    Returns:
        If train=True: features_array, label, source, feature_medians
        If train=False: features_array
    """
    signals, fields = load_signals(record)
    original_fs = fields["fs"]
    
    # Calculate expected length after resampling once (for consistency)
    expected_length = int(len(signals) * desired_sampling_rate / original_fs)
    
    # 1) Process all leads
    n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
    processed_signals = np.zeros((expected_length, n_leads))
    features_list = []
    feature_names = [
        'Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms',
        'Mean_QT_Interval_ms', 'Std_QT_Interval_ms',
        'Mean_R_Amplitude', 'Std_R_Amplitude'
    ]
    
    # If signals is single-lead, reshape for consistency
    if len(signals.shape) == 1:
        signals = signals.reshape(-1, 1)
    
    # Process each lead with NeuroKit2
    print(f"\nProcessing {n_leads} leads with NeuroKit2...")
    for i in range(n_leads):
        ecg_signal = signals[:, i]
        lead_features = {name: np.nan for name in feature_names}
        
        try:
            # 1. Normalize Signal to [-1, 1]
            normalized_signal = normalize_signal(ecg_signal)
            
            # 2. Resample using NeuroKit2 with EXACT length specified
            resampled_signal = nk.signal_resample(
                normalized_signal,
                sampling_rate=original_fs,
                desired_sampling_rate=desired_sampling_rate,
                desired_length=expected_length  # Force exact length
            )
            
            # Verify length is as expected
            if len(resampled_signal) != expected_length:
                print(f"Warning: Resampled signal length {len(resampled_signal)} doesn't match expected {expected_length}")
                # Force correct length
                if len(resampled_signal) < expected_length:
                    # Pad with edge values
                    resampled_signal = np.pad(resampled_signal, (0, expected_length - len(resampled_signal)), 'edge')
                else:
                    # Truncate
                    resampled_signal = resampled_signal[:expected_length]
            
            # 3. Clean using NeuroKit2 (applies appropriate filtering)
            cleaned_signal = nk.ecg_clean(resampled_signal, 
                                          sampling_rate=desired_sampling_rate, 
                                          method="neurokit")
            
            # Store the processed signal
            processed_signals[:, i] = cleaned_signal
            
            # 4. Find R-Peaks
            _, info = nk.ecg_peaks(cleaned_signal, sampling_rate=desired_sampling_rate)
            r_peaks = info['ECG_R_Peaks']
            
            if len(r_peaks) < 2:
                print(f"Lead {i+1}: Not enough R-peaks found. Will use median imputation.")
                features_list.append(lead_features)  # Add NaN features for now
                continue
                
            # 5. Delineate Waveforms
            try:
                _, waves_info = nk.ecg_delineate(cleaned_signal, r_peaks, 
                                                sampling_rate=desired_sampling_rate, method="dwt")
                
                # 6. Calculate Features with safety checks
                # Check if required wave points exist
                if ('ECG_S_Peaks' not in waves_info or 'ECG_Q_Peaks' not in waves_info or
                    'ECG_T_Offsets' not in waves_info):
                    print(f"Lead {i+1}: Missing required wave points. Using median imputation.")
                    features_list.append(lead_features)
                    continue
                
                # Convert lists to arrays with safety
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                t_offsets = np.array(waves_info['ECG_T_Offsets'])
                
                # Remove NaNs or invalid indices
                valid_qrs = ~np.isnan(s_peaks) & ~np.isnan(q_peaks) & (s_peaks >= q_peaks)
                valid_qt = ~np.isnan(t_offsets) & ~np.isnan(q_peaks) & (t_offsets >= q_peaks)
                
                # Calculate intervals only for valid points
                if np.any(valid_qrs):
                    qrs_durations = (s_peaks[valid_qrs] - q_peaks[valid_qrs]) / desired_sampling_rate
                    lead_features['Mean_QRS_Duration_ms'] = np.mean(qrs_durations) * 1000
                    lead_features['Std_QRS_Duration_ms'] = np.std(qrs_durations) * 1000 if len(qrs_durations) > 1 else 0
                
                if np.any(valid_qt):
                    qt_intervals = (t_offsets[valid_qt] - q_peaks[valid_qt]) / desired_sampling_rate
                    lead_features['Mean_QT_Interval_ms'] = np.mean(qt_intervals) * 1000
                    lead_features['Std_QT_Interval_ms'] = np.std(qt_intervals) * 1000 if len(qt_intervals) > 1 else 0
                
                # R-peak amplitudes
                lead_features['Mean_R_Amplitude'] = np.mean(cleaned_signal[r_peaks])
                lead_features['Std_R_Amplitude'] = np.std(cleaned_signal[r_peaks])
                
                # print(f"Lead {i+1}: Processed and features extracted successfully.")
                
            except Exception as e:
                print(f"Lead {i+1}: Error during delineation: {e}")
                # Keep NaN values for this lead
            
        except Exception as e:
            print(f"Lead {i+1}: Error during processing: {e}")
            
        features_list.append(lead_features)
    
    # Convert to numpy array (still with NaN values)
    features_array = np.array([list(d.values()) for d in features_list])
    
    # Safety check for NaN/Inf values
    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Found {nan_count} NaN values and {inf_count} Inf values in features. Imputing...")
    
    # TRAINING: Calculate and save median values for imputation
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
        
        # Impute missing values in training data
        for lead_idx in range(features_array.shape[0]):
            for j, feature_name in enumerate(feature_names):
                if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
                    features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
        
        # Save medians for test data
        feature_medians = medians_by_lead
        
    # TESTING: Apply saved median values
    elif feature_medians is not None:
        # Impute missing values in test data using training medians
        for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
            for j, feature_name in enumerate(feature_names):
                if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
                    features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
    
    # Final safety: Replace any remaining NaN/Inf with zeros
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    if train:
        label = load_label(record)
        label = np.array(label).astype(np.long)
        source = get_source(load_header(record))
        return features_array, label, source, feature_medians
    else:
        return features_array






def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    print("New submission: Training the model...")
    scaler = StandardScaler()
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    # Initialize storage for features and labels
    X_train_features = []
    y_train = []
    all_feature_medians = None  # To store feature medians for imputation

    print("Starting preprocessing...")
    for i in range(num_records):
        record = os.path.join(data_folder, records[i])
        print(f"record: {i+1} of {num_records}: {record}")

        try:
            # Unpack the returns - note we're only using features now
            features_array, label, source, feature_medians = load_and_process_signal_train(
                record, 
                desired_sampling_rate=100,
                train=True,
                feature_medians=all_feature_medians
            )
            
            print(f"features shape: {features_array.shape}, label: {label}")
            
            # Save the first record's feature medians or update
            if all_feature_medians is None:
                all_feature_medians = feature_medians
            
            # Append features - keep the original 12x6 shape
            X_train_features.append(features_array)
            y_train.append(label)
            
        except Exception as e:
            print(f"Error processing record {record}: {e}")
            continue

    # Convert lists to numpy arrays
    X_train_features = np.array(X_train_features)
    y_train = np.array(y_train)

    print(f"X_train_features shape: {X_train_features.shape}")  # Should be (num_samples, 12, 6)
    print(f"y_train shape: {y_train.shape}")

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
    
    # Improved model for ECG features
    model = ECGFeatureTransformer(
        feature_dim=feature_per_lead,
        num_leads=12,
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        dropout=DROPOUT, 
        num_classes=NUM_CLASSES
    ).to(DEVICE)
                                       
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
        
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        
    # Save model, scaler, and feature medians
    os.makedirs(model_folder, exist_ok=True)    
    save_model(model_folder, model, scaler, all_feature_medians)

    if verbose:
        print('Done.')
        print()



# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
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
    model = ECGFeatureTransformer(
        feature_dim=6,  # 6 features per lead
        num_leads=12,
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        dropout=DROPOUT, 
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Store preprocessing components with the model
    model.feature_scaler = feature_scaler
    model.feature_medians = feature_medians
    
    if verbose:
        print('Model loaded successfully.')
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
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
################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
# Save your trained model.
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