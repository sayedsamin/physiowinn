#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

#test
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # delete later
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader, TensorDataset
from helper_code import *
from scipy.signal import resample
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import random

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM=12
HIDDEN_DIM=32
N_LAYERS=1
N_HEADS=4
DROPOUT=0.1
NUM_CLASSES=2
EPOCHS = 50
LEARNING_RATE = 0.001



def filter_data(signal, lowcut=0.5, highcut=40.0, fs=250.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y




class simple_transformer_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout, num_classes, seq_length=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.seq_length = seq_length

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=self.n_layers
        )
        
        self.class_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        self.positional_encoding = nn.Parameter(torch.randn(1, int(self.seq_length + 1), self.input_dim))


        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, src):
        # src = src.permute(0, 2, 1)
        src = torch.cat((self.class_token.repeat(src.shape[0], 1, 1), src), dim=1)
        src += self.positional_encoding

        output = self.transformer_encoder(src)
        # output = output.permute(0, 2, 1)
        #use the output of the class token
        output = self.classifier(output[:, 0, :])
    
        return output
    

# Load and preprocess raw signals
# def load_and_process_signal(record, desired_length=3000):
#     signals, _ = load_signals(record)  # shape: (channels, length)

#     # Handle NaNs
#     signals = np.nan_to_num(signals, nan=0.0)

#     num_channels, signal_length = signals.shape

#     # Pad or truncate signals to desired_length
#     if signal_length < desired_length:
#         pad_width = desired_length - signal_length
#         signals = np.pad(signals, ((0,0),(0,pad_width)), 'constant')
#     else:
#         signals = signals[:, :desired_length]

#     return signals.T.astype(np.float32)  # Shape: (seq_length, num_channels)

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
    return signals, label   # Shape: (seq_length, num_channels)


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


class SignalDataset(Dataset):
    def __init__(self, data_folder, desired_samping_rate=100, low_cut=0.5, high_cut=45, desired_lenght=7):
        self.records = find_records(data_folder)
        self.data_folder = data_folder
        self.desired_samping_rate = desired_samping_rate
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.desired_lenght = desired_lenght

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_folder, self.records[idx])
        #this parameter is fixed for now 
        signals, label = load_and_process_signal_train(record_path, 
                                                desired_samping_rate=self.desired_samping_rate, 
                                                low_cut=self.low_cut, high_cut=self.high_cut, 
                                                desired_lenght=self.desired_lenght)
        
        signals = torch.tensor(signals, dtype=torch.float32)  # Convert to PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)  # Convert to PyTorch tensor
        return signals, label

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    scaler = StandardScaler()
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    #loading functions with fixed parameters
    X_train = []
    y_train = []
    for i in range(num_records):
        record = os.path.join(data_folder, records[i])
        signals, label = load_and_process_signal_train(record, desired_samping_rate=100, low_cut=0.5, high_cut=45, desired_lenght=7)
        X_train.append(signals)
        y_train.append(label)
        

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # print(y_train)
    # in y_train, calculate the number of 1s and 0s
    num_1s = np.sum(y_train == 1)
    num_0s = np.sum(y_train == 0)
    print(f"Number of 1s: {num_1s}")
    print(f"Number of 0s: {num_0s}")

    # scale the data
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    
    print(scaler.mean_, scaler.scale_)

    # print("X_train shape: ", X_train.shape)

    #create tensor dataset
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Get num_channels from the first sample
    sample_signals, _ = dataset[0]
    num_channels = sample_signals.shape[1]
    seq_length = sample_signals.shape[0]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = simple_transformer_encoder(input_dim=INPUT_DIM,
                                       seq_length=seq_length, 
                                        hidden_dim=HIDDEN_DIM, 
                                        n_layers=N_LAYERS, 
                                        n_heads=N_HEADS, 
                                        dropout=DROPOUT, 
                                        num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            
            # Forward pass
            outputs = model(batch_signals)
            
            # Compute loss
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item() * batch_signals.size(0)

            # Store predictions and true labels
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
        # print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}')
        
# Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    
    save_model(model_folder, model, seq_length, num_channels, scaler)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
# def load_model(model_folder, verbose):
#     model_filename = os.path.join(model_folder, 'model.sav')
#     model = joblib.load(model_filename)
#     return model

# Load your Transformer model
def load_model(model_folder, verbose):
    checkpoint = torch.load(os.path.join(model_folder, 'transformer_model.pt'), 
                              map_location=DEVICE, weights_only=False)

    if verbose:
        print('Extracting features and labels from the data...')

    model = simple_transformer_encoder(input_dim=INPUT_DIM, 
                                       hidden_dim=HIDDEN_DIM, 
                                       n_layers=N_LAYERS, 
                                       n_heads=N_HEADS, 
                                       dropout=DROPOUT, 
                                       num_classes=NUM_CLASSES,
                                       seq_length=700).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract the scaler from the checkpoint
    model.scaler = checkpoint.get('scaler', None)
    if model.scaler is None and verbose:
        print('Scaler not found in the checkpoint.')

    return model

# Change made by samin 
    # return {
    #     'model': model,
    #     'scaler': scaler 
    # }



# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
# def run_model(record, model, verbose):
#     # Load the model.
#     model = model['model']

#     # Extract the features.
#     features = extract_features(record)
#     features = features.reshape(1, -1)

#     # Get the model outputs.
#     binary_output = model.predict(features)[0]
#     # probability_output = model.predict_proba(features)[0][1]
#     probability_output = model.predict_proba(features)[0]


#     return binary_output, probability_output


# Run inference on raw signals
def run_model(record, model, verbose):
    # print(model)
    # check data type of model

 
    # Load the model.
    # model = model['model']
    # scaler = model['scaler']

    # Extract the features.

    # signals, fields = load_and_process_signal(record, desired_samping_rate=100, low_cut=0.5, high_cut=45, desired_lenght=7)
    
    signals = load_and_process_signal_test(record, desired_samping_rate=100, low_cut=0.5, high_cut=45, desired_lenght=7)


    if len(signals.shape) > 2:
        signals = model.scaler.transform(signals.reshape(signals.shape[0], -1)).reshape(signals.shape)  
    else:
        signals = model.scaler.transform(signals.reshape(1, -1)).reshape(signals.shape)
    signals = torch.tensor(signals).unsqueeze(0).to(DEVICE)  # shape: (1, seq_length, channels)
    

    with torch.no_grad():
        logits = model(signals)
        
        probs = F.softmax(logits, dim=1).cpu().numpy()

        pred_class = int(np.argmax(probs))

    binary_output = bool(pred_class)
    
    #just atking trh prob for "changas" class
    probability_output = probs[0,1]

    
    print(f"pred_class: {pred_class}, binary_output: {binary_output}, probability_output: {probability_output}")

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
# def save_model(model_folder, model):
#     d = {'model': model}
#     filename = os.path.join(model_folder, 'model.sav')
#     joblib.dump(d, filename, protocol=0)


# Save model function
def save_model(model_folder, model, seq_length, num_channels, scaler):
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_length': seq_length,
        'num_channels': num_channels,
        'scaler': scaler
    }, os.path.join(model_folder, 'transformer_model.pt'))

# Shyamal's
# def save_model(model_folder, model, seq_length, num_channels, scaler):
# torch.save({
#         'model_state_dict': model.state_dict(),
#         'seq_length': seq_length,
#         'num_channels': num_channels,
#         'scaler': scaler
#     }, os.path.join(model_folder, 'transformer_model.pt'))