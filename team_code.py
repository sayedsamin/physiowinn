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
EPOCHS = 50
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


class simple_transformer_encoder(nn.Module):
    def __init__(self, input_dim, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT, num_classes=NUM_CLASSES):
        super(simple_transformer_encoder, self).__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.n_heads, dropout=self.dropout, batch_first=True),
            num_layers=N_LAYERS
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, 700, self.input_dim))  # Assuming seq_length is 120
        self.fc = nn.Linear(self.input_dim, self.num_classes)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.size()
        # Add positional encoding
        # x = x + self.positional_encoding[:, :seq_length, :].expand(batch_size, -1, -1)
        # Apply dropout
        # x = self.dropout(x)
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        # Take the mean across the sequence length
        x = x.mean(dim=1)

        x = self.fc(x)  # Final linear layer
        
        return x


def load_and_process_signal_train(record, desired_samping_rate, low_cut = 0.5, high_cut = 45, desired_lenght=7):
    signals, fields = load_signals(record) #load the data
    # print(fields)
    original_sampling_rate = fields["fs"] #original sampling rate
    # label = 1 if "True" in fields["comments"][2] else 0 #label is 1 if the comment contains "TRUE" else 0
    label = load_label(record)


    
    if original_sampling_rate != desired_samping_rate: 
        num_samples = int(len(signals) * desired_samping_rate / original_sampling_rate)
        signals = resample(signals, num_samples, axis=0) # Resample the signal

    # Filter the signal
    signals = filter_data(signals, lowcut=low_cut, highcut=high_cut, fs=desired_samping_rate) # Filter the signal

    signals = np.nan_to_num(signals, nan=0.0) # good for any missing values but probably there wont be any

    signal_length, _  = signals.shape #get the length of the signal
    
    total_samples_to_be_considered = desired_samping_rate * desired_lenght #total samples to be considered

    if signal_length > total_samples_to_be_considered: #cut the signal 
        signals = signals[:total_samples_to_be_considered, :] 
    elif signal_length < total_samples_to_be_considered: #repeat the signals from the start
        pad_width = total_samples_to_be_considered - signal_length
        signals = np.pad(signals, ((0, pad_width), (0, 0)), mode='wrap')

    # change data type to float32 and long
    signals = signals.astype(np.float32) #change the data type to float32
    label = np.array(label).astype(np.long) #change the data type to long
    header = load_header(record)    
    source = get_source(header)
    return signals, label, source   # Shape: (seq_length, num_channels)


def load_and_process_signal_test(record, desired_samping_rate, low_cut = 0.5, high_cut = 45, desired_lenght=7):
    signals, fields = load_signals(record) #load the data
    # print(fields)
    original_sampling_rate = fields["fs"] #original sampling rate
    # label = 1 if "True" in fields["comments"][2] else 0 #label is 1 if the comment contains "TRUE" else 0
    # label = load_label(record)

    
    if original_sampling_rate != desired_samping_rate: 
        num_samples = int(len(signals) * desired_samping_rate / original_sampling_rate)
        signals = resample(signals, num_samples, axis=0) # Resample the signal

    # Filter the signal
    signals = filter_data(signals, lowcut=low_cut, highcut=high_cut, fs=desired_samping_rate) # Filter the signal

    signals = np.nan_to_num(signals, nan=0.0) # good for any missing values but probably there wont be any

    signal_length, _  = signals.shape #get the length of the signal
    
    total_samples_to_be_considered = desired_samping_rate * desired_lenght #total samples to be considered

    if signal_length > total_samples_to_be_considered: #cut the signal 
        signals = signals[:total_samples_to_be_considered, :] 
    elif signal_length < total_samples_to_be_considered: #repeat the signals from the start
        pad_width = total_samples_to_be_considered - signal_length
        signals = np.pad(signals, ((0, pad_width), (0, 0)), mode='wrap')

    # change data type to float32 and long
    signals = signals.astype(np.float32) #change the data type to float32
    # label = np.array(label).astype(np.long) #change the data type to long    
    return signals   # Shape: (seq_length, num_channels)


def windowing(data, window_size, overlap):
    # ensure array
    data = np.asarray(data)
    # print(f"data shape: {data.shape}")
    n_samples, n_features = data.shape
    # print(f"n_samples: {n_samples}, n_features: {n_features}")
    if window_size > n_samples:
        raise ValueError(f"window_size ({window_size}) > n_samples ({n_samples})")
    if not (0 <= overlap < window_size):
        raise ValueError(f"overlap must be ≥0 and < window_size ({window_size})")

    step = window_size - overlap
    # number of full windows
    n_windows = 1 + (n_samples - window_size) // step
    # print(f"n_windows: {n_windows}, n_samples: {n_samples}, window_size: {window_size}, step: {step}")

    windows = np.empty((n_windows, window_size, n_features), dtype=data.dtype)
    for i in range(n_windows):
        start = i * step
        windows[i] = data[start : start + window_size]

    return windows




# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    print("New submission: Training the model...")
    scaler = StandardScaler()
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    #loading functions with fixed parameters
    X_train = []
    y_train = []

    print ("Starting preprocessing...")
    for i in range(num_records):
        record = os.path.join(data_folder, records[i])
        print(f"record: {i} of {num_records}: {record}")


        signals, label, source = load_and_process_signal_train(record, desired_samping_rate=100, low_cut=0.5, high_cut=45, desired_lenght=7)
        print(f"signals shape: {signals.shape}, label: {label}, source: {source}")
        print(signals.shape)


        # try:
        #     if source == 'CODE-15%':
        #         print(f"Skipping record {record} due to source CODE-15%")
        #         continue
        # except Exception as e:
        #     print(f"Error processing record {record}: {e}")
        #     continue

        

        X_train.append(signals)
        y_train.append(label)

        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # else:
    #     print ("Loading the data from the saved files...")
    #     X_train = np.load(os.path.join(data_folder, "X_train.npy"))
    #     y_train = np.load(os.path.join(data_folder, "y_train.npy"))


    print("Starting scaling...")
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    
  


   
    #create tensor dataset
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = simple_transformer_encoder(input_dim=12,
                                        n_heads=N_HEADS, 
                                        n_layers=N_LAYERS, 
                                        dropout=DROPOUT, 
                                        num_classes=NUM_CLASSES).to(DEVICE)
                                       
    
    num_0s = np.sum(y_train == 0)
    num_1s = np.sum(y_train == 1)

    class_weights_0 = torch.tensor([1.0, num_0s/num_1s], dtype=torch.float32).to(DEVICE)

    focal_loss = FocalLoss(gamma=2, alpha=class_weights_0, reduction='mean', task_type='multi-class', num_classes=NUM_CLASSES).to(DEVICE)
    criterion = focal_loss.to(DEVICE)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # epochs = 20

    if verbose:
        print('Starting training...')

    model.train()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_signals, batch_labels in dataloader:
            batch_signals = batch_signals.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_signals)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_signals.size(0)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())  # Move to CPU and convert to numpy
            all_labels.extend(batch_labels.cpu().numpy())     # Move to CPU and convert to numpy

        # Compute average loss for the epoch
        epoch_loss /= len(dataset)
        # Calculate accuracy
        correct = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()
        accuracy = correct / len(all_labels)
        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='macro')  # You can use 'macro' or 'micro' as well
        
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        
    os.makedirs(model_folder, exist_ok=True)    
    save_model(model_folder, model, scaler)

    if verbose:
        print('Done.')
        print()



# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    checkpoint = torch.load(os.path.join(model_folder, 'transformer_model.pt'), 
                              map_location=DEVICE, weights_only=False)

    if verbose:
        print('Extracting features and labels from the data...')
    model = simple_transformer_encoder(input_dim=12,
                                        n_heads=N_HEADS, 
                                        n_layers=N_LAYERS, 
                                        dropout=DROPOUT, 
                                        num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.scaler = checkpoint.get('scaler', None)
    if model.scaler is None and verbose:
        print('Scaler not found in the checkpoint.')
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    signals = load_and_process_signal_test(record, desired_samping_rate=100, low_cut=0.5, high_cut=45, desired_lenght=7)
    

    if len(signals.shape) > 2:
        signals_features = model.scaler.transform(signals.reshape(signals.shape[0], -1)).reshape(signals.shape)  
    else:
        signals_features = model.scaler.transform(signals.reshape(1, -1)).reshape(signals.shape)
    signals_features = torch.tensor(signals_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape: (1, seq_length, channels)
    

    with torch.no_grad():
        logits = model(signals_features)
        
        probs = F.softmax(logits, dim=1).cpu().numpy()

        pred_class = int(np.argmax(probs))

    binary_output = bool(pred_class)
    
    #just atking trh prob for "changas" class
    probability_output = probs[0,1]

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
def save_model(model_folder, model, scaler):
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'seq_length': seq_length,
        # 'num_channels': num_channels,
        'scaler': scaler
    }, os.path.join(model_folder, 'transformer_model.pt'))