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
import scipy
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

from torch.optim.lr_scheduler import OneCycleLR

import warnings
warnings.filterwarnings("ignore")


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    # PyTorch ≥1.8: force deterministic algorithms
    try:
        torch.use_deterministic_algorithms(False)
    except AttributeError:
        pass


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

LEARNING_RATE = 0.0001
DROPOUT = 0.2
NUM_CLASSES = 2  
EPOCHS = 300

def filter_data(signal, lowcut=0.5, highcut=40.0, fs=250.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


# class gnn_model(nn.Module):
#     def __init__(self, input_dim=30, num_leads=12, num_classes=2):
#         super(gnn_model, self).__init__()

#         # temp = input_dim
#         # input_dim = num_leads
#         # num_leads = temp

#         self.input_dim = input_dim
#         self.num_leads = num_leads
#         self.num_classes = num_classes

#         self.adj_linear = nn.Parameter(torch.rand(num_leads, input_dim), requires_grad=True)
   
#         self.activation = nn.SELU() # nn.LeakyReLU() #  nn.GELU() #  nn.GELU() # nn.LeakyReLU() #
#         #make conv1 as a sequential layer
#         expand = input_dim*10
#         self.gnn1 = nn.Sequential( nn.Linear(input_dim, expand  ),
                   
#             self.activation,
#               nn.LayerNorm(  expand ),
#               nn.Dropout(0.2)
     
#         )

#         # self.cnn = conv_model( input_dim=expand, num_leads=num_leads, num_classes=num_classes)

#         # self.gnn2 = nn.Sequential( nn.Linear(expand, expand  ),
#         #                         self.activation,
#         #       nn.LayerNorm(  expand ),
#         #       nn.Dropout(0.2)
     
#         # )

#         # self.linear1 = nn.Sequential( nn.Linear(expand, expand,  ),
#         #                              nn.LayerNorm(expand),
#         #     self.activation,
#         #     nn.Dropout(0.2)
     
#         # )

#         # self.linear2 = nn.Sequential(   nn.Linear(expand , expand,  ),
#         #                              nn.LayerNorm(expand),
#         #     self.activation,
#         #     nn.Dropout(0.2)
     
#         # )

#         units = num_leads*expand#
#         unit_out = 128 # 64
#         # Fully connected layers
#         self.layer_norm = nn.LayerNorm(  units )  # Adjust input size based on conv output
#         self.fc1 = nn.Linear(units, unit_out )  # Adjust input size based on conv output
#         self.fc2 = nn.Linear(unit_out, num_classes)
#         self.dropout = nn.Dropout(0.7)
#         self.flatten = nn.Flatten()

   
#     def forward(self, x):

#         # x = x.permute(0, 2, 1)  # Change shape to (batch_size,  input_dim, num_leads )

#         # print("4444444444444444444444")
#         # print(x.shape)

#         # print(f"Input shape: {x.shape}")  # Debugging line to check input shape


#         ### calculating adjencney matrix
#         a = x #* self.adj_linear
   
#         # do sigmoid
#         # a =  self.activation(a )  # Normalize across features
#         #a = torch.sigmoid(a)

#         # .activation( a )

#         batch, C, F = x.shape

#         a_mean = torch.mean(a, dim=-1, keepdims=True)
#         a_std = torch.std(a, unbiased=True, dim=-1, keepdims=True)

#         a_centered = a -a_mean

   
#         # 3. covariance
#         #    (use F−1 if you set unbiased=True above so that denominator matches)
#         # cov = a_centered @ a_centered.transpose(1, 2) / (F - 1)   # → [batch, C, C]

#         cov = (a_centered.cpu() @ a_centered.transpose(1, 2).cpu()) / (F - 1)
   

#         # 4. outer product of stds
#         var_outer = a_std.cpu() @ a_std.transpose(1, 2).cpu()               # → [batch, C, C]

#         # 5. correlation matrix
#         corr = cov / (var_outer + 1e-8)                              # → [batch, C, C]

#         corr = corr.to(x.device)  
#         # print(torch.min(corr) )
#         # corr     = corr - corr.min()
#         corr     = torch.relu(corr ) #+ torch.eye(12).to(DEVICE) # Normalize to [0, 1]
#         # print(corr)


#         corr = corr/( torch.sum(corr, dim=-1, keepdim=True) +1e-6 ) # Normalize across features

     

#         # # 6. graph‐Laplace normalization
#         deg = torch.sum(corr, axis=-1)                              # → [batch, C]

#         norm_deg = torch.diag_embed( 1.0 / torch.sqrt(deg) + 1e-8 )

   
#         norm_adj = torch.matmul(norm_deg, torch.matmul(corr, norm_deg) )


#         A_k = torch.linalg.matrix_power(norm_adj, 1 )

#         # print( A_k )

#         z = torch.matmul( A_k, x )
#         x = self.gnn1(z) # batch size x channels x features

#         # x = self.linear1(x)

#         # x = self.cnn(x)

#         # # x = torch.matmul( A_k, x )
#         # # x = self.gnn2(x) # batch size x channels x features


#         # # # # # second gnn layer
#         # z = torch.matmul( norm_adj, x )
#         # x = self.gnn2(z)

#         # # x = self.linear1(x) # batch size x channels x features
#         # # x = self.linear2(x) # batch size x channels x features

   




#         # #x, _ = torch.max(x, dim=1)  # Aggregate across leads (channels)
   

#         # ## x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, num_leads, input_dim)
#         # ## x = self.conv1(x)
#         # ## x = self.conv2(x)
   
#         x = self.flatten(x)  # Flatten the output for fully connected layers
   
#         x = self.layer_norm(x)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
   
#         return x
   
# class transformer(nn.Module):
#     def __init__(self, input_dim=30, num_leads=12, num_classes=2):
#         super(transformer, self).__init__()

#         # temp = input_dim
#         # input_dim = num_leads
#         # num_leads = temp

#         self.input_dim = input_dim
#         self.num_leads = num_leads
#         self.num_classes = num_classes

#         self.activation = nn.ReLU()

#         self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4,
#                                            batch_first=True, dim_feedforward = 2*input_dim , activation=self.activation )

#    # # )

#         units = num_leads*input_dim #
#         unit_out = 128 # 64
#         # Fully connected layers
#         self.layer_norm = nn.LayerNorm(  units )  # Adjust input size based on conv output
#         self.fc1 = nn.Linear(units, unit_out )  # Adjust input size based on conv output
#         self.fc2 = nn.Linear(unit_out, num_classes)
#         self.dropout = nn.Dropout(0.7)
#         self.flatten = nn.Flatten()

   
#     def forward(self, x):

   
#         x = self.transformer(x)
   
#         x = self.flatten(x)  # Flatten the output for fully connected layers
   
#         x = self.layer_norm(x)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
   
#         return x

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
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
     
        )
        self.conv2 = nn.Sequential(
            self.conv2_layer,
            nn.ReLU(),
            nn.BatchNorm2d(64),
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
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val > 0:
        # Formula: -1 + 2 * (x - min) / (max - min)
        return -1 + 2 * (signal - min_val) / (max_val - min_val)
    else:
        # Handle the case of a flat signal to avoid division by zero
        return np.zeros_like(signal)

 


def process_single_lead(lead_data, lead_idx, original_fs, desired_sampling_rate, expected_length,
                        feature_config=None):
    """
    Process a single lead and extract requested ECG features with modular selection.
   
    Args:
        lead_data: Signal data for one lead
        lead_idx: Index of the lead
        original_fs: Original sampling frequency
        desired_sampling_rate: Target sampling frequency
        expected_length: Expected length after resampling
        feature_config: Dictionary of feature groups to enable/disable:
            {
                'basic_intervals': True,      # QRS duration, QT interval
                'amplitude': True,        # R-peak, P-wave amplitudes
                'hrv': True,                  # Heart rate variability metrics
                'p_wave': True,               # P wave features
                'st_segment': True,           # ST segment analysis
                't_wave': True,               # T wave morphology
                'pr_interval': True,          # PR interval analysis
                'qtc': True,                  # Corrected QT interval
                'qrs_fragmentation': True     # QRS fragmentation detection
            }
   
    Returns:
        Tuple of (lead_idx, lead_features)
    """
    # Default feature configuration - enable all features
    if feature_config is None:
        feature_config = {
            'basic_intervals': True,   # Keep QRS, QT intervals
            'amplitude': True,         # Keep R-peak amplitudes
            'hrv': True,
            'advanced_hrv': False,               # Keep heart rate variability metrics
            'p_wave': True,        # Keep P wave features
            'st_segment': False,       # Disable ST segment analysis
            't_wave': False,           # Disable T wave features
            'pr_interval': False,      # Disable PR interval features
            'qtc': False,              # Disable QTc features
            'qrs_fragmentation': False, # Disable QRS fragmentation detection
            'hrv_non_linear': False,  # Enable non-linear HRV features
            'hrv_welch': False,  # Enable Welch's method for HRV

        }


   
   
    # Initialize feature dictionary based on enabled features
    feature_names = []
   
    # Basic interval features
    if feature_config.get('basic_intervals', True):
        feature_names.extend(['Mean_QRS_Duration_ms', 'Std_QRS_Duration_ms',
                             'Mean_QT_Interval_ms', 'Std_QT_Interval_ms'])
   
    # Amplitude features
    if feature_config.get('amplitude', True):
        feature_names.extend(['Mean_R_Amplitude', 'Std_R_Amplitude', 'QRS_Net_Deflection'])
   
    # HRV features
    if feature_config.get('hrv', True):
        feature_names.extend(['RR_Interval_RMSSD_ms'])

    if feature_config.get('hrv_non_linear', True):
        feature_names.extend(['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'RR_SDNN_ms', 'pNN50'])

    if feature_config.get('hrv_welch', True):
        feature_names.extend(['HRV_LF', 'HRV_HF', 'HRV_LF_norm', 'HRV_LF_HF'])

 
    # P wave features
    if feature_config.get('p_wave', True):
        feature_names.extend(['Mean_P_Amplitude', 'Std_P_Amplitude',
                             'Mean_P_Duration_ms', 'Std_P_Duration_ms'])
   
    # ST segment features
    if feature_config.get('st_segment', True):
        feature_names.extend(['ST_Deviation', 'ST_Deviation_Std'])
   
    # T wave features
    if feature_config.get('t_wave', True):
        feature_names.extend(['Mean_T_Amplitude', 'T_Wave_Inversion_Ratio'])
   
    # PR interval features
    if feature_config.get('pr_interval', True):
        feature_names.extend(['Mean_PR_Interval_ms', 'Std_PR_Interval_ms'])
   
    # QTc features
    if feature_config.get('qtc', True):
        feature_names.extend(['QTc_Bazett_ms', 'QTc_Fridericia_ms'])
   
    # QRS fragmentation
    if feature_config.get('qrs_fragmentation', True):
        feature_names.extend(['QRS_Fragmentation_Ratio'])

    # Initialize features dictionary - making then nan by default
    lead_features = {name: np.nan for name in feature_names}

    try:
        # 1. Fast normalization
        normalized_signal = normalize_signal(lead_data)
   
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
        peaks_hrv, info = nk.ecg_peaks(
            cleaned_signal,
            sampling_rate=desired_sampling_rate,
            method="pantompkins"
        )

        r_peaks = info['ECG_R_Peaks']
        # print(f"Lead {lead_idx} - Number of R-peaks detected: {len(r_peaks)}")
        if len(r_peaks) < 2:
            return lead_idx, lead_features

        # print(np.sum(peaks_hrv,axis=0))

        # print(np.isnan(peaks_hrv).any(), np.isnan(info['ECG_R_Peaks']).any() )
        if feature_config.get('hrv_non_linear', True):
            hrv_nonlinear = nk.hrv_nonlinear(peaks_hrv, sampling_rate=desired_sampling_rate)#, psd_method="welch", normalize=False )


            lead_features['HRV_SD1'] = hrv_nonlinear['HRV_SD1'].values[0]
            lead_features['HRV_SD2'] = hrv_nonlinear['HRV_SD2'].values[0]
            lead_features['HRV_SD1SD2'] = hrv_nonlinear['HRV_SD1SD2'].values[0]


        if feature_config.get('hrv_welch', True):

            peaks_location  = peaks_hrv.loc[ peaks_hrv['ECG_R_Peaks']==1].index  
            time_peaks = peaks_location / desired_sampling_rate  # seconds
            rr_intervals = np.diff(time_peaks)  # in seconds - duration between R-peaks

            rr_times = time_peaks[:-1] + rr_intervals / 2
            # Choose interpolation frequency (e.g., 4 Hz)
            fs_interp = 4  # samples per second
            t_interp = np.linspace(rr_times[0], rr_times[-1], int((rr_times[-1] - rr_times[0]) * fs_interp))

            # Interpolate RR intervals
            f_interp =  scipy.interpolate.interp1d(rr_times, rr_intervals, kind='cubic', fill_value="extrapolate")
            rr_uniform = f_interp(t_interp)

            frequencies, psd = welch(rr_uniform, fs=fs_interp, nperseg=min(256, len(rr_uniform)))

            # LF and HF bands
            lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
            hf_band = (frequencies >= 0.15) & (frequencies <= 0.4)

            df = frequencies[1] - frequencies[0]  # assumes uniform spacing

            lf_power = np.sum( psd[lf_band] )*df
            hf_power = np.sum( psd[hf_band] )*df

            lead_features['HRV_LF'] = lf_power
            lead_features['HRV_HF'] = hf_power
            lead_features['HRV_LF_norm'] = lf_power /( 1e-6 + lf_power + hf_power )
            lead_features['HRV_LF_HF'] = lf_power /( 1e-6 + hf_power )
            # print('done spectral')


        # Extract HRV features
        if feature_config.get('hrv', True):

            if len(r_peaks) >= 3:
                rr_intervals = np.diff(r_peaks) / desired_sampling_rate  # in seconds
                # Filter physiologically reasonable RR intervals (0.3s to 2.0s)
                valid_rr = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
    
                if len(valid_rr) >= 2:
                    # RMSSD - Root Mean Square of Successive Differences
                    rr_diff = np.diff(valid_rr * 1000)  # Convert to ms
                    lead_features['RR_Interval_RMSSD_ms'] = np.sqrt(np.mean(rr_diff**2))
         
                    # Advanced HRV metrics
                    if feature_config.get('advanced_hrv', True):
                        if len(valid_rr) >= 5:
                            # SDNN - Standard deviation of NN intervals
                            lead_features['RR_SDNN_ms'] = np.std(valid_rr * 1000)
                            # pNN50 - Percentage of successive RR intervals that differ by more than 50ms
                            differences = np.abs(np.diff(valid_rr * 1000))
                            lead_features['pNN50'] = 100 * np.sum(differences > 50) / len(differences)
                        else:
                            lead_features['RR_SDNN_ms'] = np.nan
                            lead_features['pNN50'] = np.nan
                
        # print('done hrv')
        # 5. Delineate waveforms
    
        _, waves_info = nk.ecg_delineate(
            cleaned_signal,
            r_peaks,
            sampling_rate=desired_sampling_rate,
            method="dwt"
        )


        # Basic interval features
        if feature_config.get('basic_intervals', True):
            # QRS Duration features
            if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info:
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
           
                # Calculate QRS durations directly, handling NaNs
                qrs_durations = (s_peaks - q_peaks) / desired_sampling_rate
                valid_qrs = ~np.isnan(qrs_durations) & (qrs_durations > 0)
                if np.any(valid_qrs):
                    valid_qrs_durations = qrs_durations[valid_qrs]
                    lead_features['Mean_QRS_Duration_ms'] = np.nanmean(valid_qrs_durations) * 1000
                    lead_features['Std_QRS_Duration_ms'] = np.nanstd(valid_qrs_durations) * 1000 if len(valid_qrs_durations) > 1 else 0
      
            # QT Interval features
            if 'ECG_T_Offsets' in waves_info and 'ECG_Q_Peaks' in waves_info:
                t_offsets = np.array(waves_info['ECG_T_Offsets'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
         
                qt_intervals = (t_offsets - q_peaks) / desired_sampling_rate
                valid_qt = ~np.isnan(qt_intervals) & (qt_intervals > 0)
                if np.any(valid_qt):
                    valid_qt_intervals = qt_intervals[valid_qt]
                    lead_features['Mean_QT_Interval_ms'] = np.nanmean(valid_qt_intervals) * 1000
                    lead_features['Std_QT_Interval_ms'] = np.nanstd(valid_qt_intervals) * 1000 if len(valid_qt_intervals) > 1 else 0
           
                    # QTc Calculation
                    if feature_config.get('qtc', True) and len(r_peaks) >= 2:
                        mean_rr = np.mean(np.diff(r_peaks)) / desired_sampling_rate
                        mean_qt_sec = lead_features['Mean_QT_Interval_ms'] / 1000
               
                        # Bazett's formula: QTc = QT / √RR
                        lead_features['QTc_Bazett_ms'] = (mean_qt_sec / np.sqrt(mean_rr)) * 1000
               
                        # Fridericia's formula: QTc = QT / ∛RR
                        lead_features['QTc_Fridericia_ms'] = (mean_qt_sec / np.cbrt(mean_rr)) * 1000
       
        # Amplitude features
        if feature_config.get('amplitude', True):
            # R-peak amplitude features
            if len(r_peaks) > 0:
                lead_features['Mean_R_Amplitude'] = np.nanmean(cleaned_signal[r_peaks])
                lead_features['Std_R_Amplitude'] = np.nanstd(cleaned_signal[r_peaks])
     
            # QRS Net Deflection
            if 'ECG_S_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info and len(r_peaks) > 0:
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
        
                # Remove NaN values
                valid_s = s_peaks[~np.isnan(s_peaks)]
                valid_q = q_peaks[~np.isnan(q_peaks)]
       
                if len(valid_s) > 0 and len(valid_q) > 0:

                    valid_s_indices = valid_s.astype(int)
                    valid_q_indices = valid_q.astype(int)
             
                    # Ensure indices are within bounds
                    valid_s_indices = valid_s_indices[ valid_s_indices < len(cleaned_signal) ]
                    valid_q_indices = valid_q_indices[ valid_q_indices < len(cleaned_signal) ]
            
                    if len(valid_s_indices) > 0 and len(valid_q_indices) > 0:
                        s_amplitudes = cleaned_signal[ valid_s_indices ]
                        q_amplitudes = cleaned_signal[ valid_q_indices ]
                        r_amplitudes = cleaned_signal[ r_peaks.astype(int) ]
                  
                        mean_qs = (np.nanmean(q_amplitudes) + np.nanmean(s_amplitudes)) / 2
                        qrs_net_deflection = np.nanmean(r_amplitudes) - mean_qs
                        lead_features['QRS_Net_Deflection'] = qrs_net_deflection
          
        # P wave features
        if feature_config.get('p_wave', True):
            # P Wave Amplitude Features
            if 'ECG_P_Peaks' in waves_info:
                p_peaks = np.array(waves_info['ECG_P_Peaks'])
            
                valid_p = ~np.isnan(p_peaks)
                if np.any(valid_p):
                    valid_p_indices = p_peaks[valid_p].astype(int)
                    valid_p_indices = valid_p_indices[(valid_p_indices >= 0) & (valid_p_indices < len(cleaned_signal))]
            
                    if len(valid_p_indices) > 0:
                        p_amplitudes = cleaned_signal[valid_p_indices]
                        lead_features['Mean_P_Amplitude'] = np.nanmean(p_amplitudes)
                        lead_features['Std_P_Amplitude'] = np.nanstd(p_amplitudes) if len(p_amplitudes) > 1 else 0.0
        
            # P Wave Duration
            if 'ECG_P_Onsets' in waves_info and 'ECG_P_Offsets' in waves_info:
                p_onsets = np.array(waves_info['ECG_P_Onsets'])
                p_offsets = np.array(waves_info['ECG_P_Offsets'])
           
                valid_onsets = ~np.isnan(p_onsets)
                valid_offsets = ~np.isnan(p_offsets)
          
                if np.any(valid_onsets) and np.any(valid_offsets):
                    valid_on_indices = p_onsets[valid_onsets].astype(int)
                    valid_off_indices = p_offsets[valid_offsets].astype(int)
               
                    valid_on_indices = valid_on_indices[(valid_on_indices >= 0) & (valid_on_indices < len(cleaned_signal))]
                    valid_off_indices = valid_off_indices[(valid_off_indices >= 0) & (valid_off_indices < len(cleaned_signal))]
              
                    p_durations = []
                    for on_idx in valid_on_indices:
                        following_offs = valid_off_indices[valid_off_indices > on_idx]
                        if len(following_offs) > 0:
                            p_durations.append((following_offs[0] - on_idx) / desired_sampling_rate)
                
                    if len(p_durations) > 0:
                        lead_features['Mean_P_Duration_ms'] = np.nanmean(p_durations) * 1000
                        lead_features['Std_P_Duration_ms'] = np.nanstd(p_durations) * 1000 if len(p_durations) > 1 else 0.0
           
        # ST segment analysis
        if feature_config.get('st_segment', True):
            if 'ECG_S_Peaks' in waves_info and 'ECG_T_Onsets' in waves_info:
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
                t_onsets = np.array(waves_info['ECG_T_Onsets'])
           
                valid_s = ~np.isnan(s_peaks)
                valid_t = ~np.isnan(t_onsets)
        
                if np.any(valid_s) and np.any(valid_t):
                    valid_s_indices = s_peaks[valid_s].astype(int)
                    valid_t_indices = t_onsets[valid_t].astype(int)
                
                    valid_s_indices = valid_s_indices[(valid_s_indices >= 0) & (valid_s_indices < len(cleaned_signal))]
                    valid_t_indices = valid_t_indices[(valid_t_indices >= 0) & (valid_t_indices < len(cleaned_signal))]
             
                    st_segments = []
                    for s_idx in valid_s_indices:
                        following_t = valid_t_indices[valid_t_indices > s_idx]
                        if len(following_t) > 0:
                            st_level = np.mean(cleaned_signal[s_idx:following_t[0]])
                            st_segments.append(st_level)
               
                    if st_segments:
                        lead_features['ST_Deviation'] = np.mean(st_segments)
                        lead_features['ST_Deviation_Std'] = np.std(st_segments) if len(st_segments) > 1 else 0
            
        # T wave morphology
        if feature_config.get('t_wave', True):
            if 'ECG_T_Peaks' in waves_info:
                t_peaks = np.array(waves_info['ECG_T_Peaks'])
   
                valid_t_peaks = ~np.isnan(t_peaks)
                if np.any(valid_t_peaks):
                    valid_t_indices = t_peaks[valid_t_peaks].astype(int)
                    valid_t_indices = valid_t_indices[(valid_t_indices >= 0) & (valid_t_indices < len(cleaned_signal))]
       
                    if len(valid_t_indices) > 0:
                        t_amplitudes = cleaned_signal[valid_t_indices]
                        lead_features['Mean_T_Amplitude'] = np.nanmean(t_amplitudes)
                        lead_features['T_Wave_Inversion_Ratio'] = np.mean(t_amplitudes < 0)
   
        # PR interval analysis
        if feature_config.get('pr_interval', True):
            if 'ECG_P_Peaks' in waves_info and 'ECG_Q_Peaks' in waves_info:
                p_peaks = np.array(waves_info['ECG_P_Peaks'])
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
   
                pr_intervals = []
                valid_p = ~np.isnan(p_peaks)
                valid_q = ~np.isnan(q_peaks)
   
                if np.any(valid_p) and np.any(valid_q):
                    valid_p_indices = p_peaks[valid_p].astype(int)
                    valid_q_indices = q_peaks[valid_q].astype(int)
       
                    for p_idx in valid_p_indices:
                        following_q = valid_q_indices[valid_q_indices > p_idx]
                        if len(following_q) > 0:
                            pr_interval = (following_q[0] - p_idx) / desired_sampling_rate * 1000  # in ms
                            pr_intervals.append(pr_interval)
       
                    if pr_intervals:
                        lead_features['Mean_PR_Interval_ms'] = np.mean(pr_intervals)
                        lead_features['Std_PR_Interval_ms'] = np.std(pr_intervals) if len(pr_intervals) > 1 else 0
   
        # QRS fragmentation detection
        if feature_config.get('qrs_fragmentation', True):
            if 'ECG_Q_Peaks' in waves_info and 'ECG_S_Peaks' in waves_info:
                q_peaks = np.array(waves_info['ECG_Q_Peaks'])
                s_peaks = np.array(waves_info['ECG_S_Peaks'])
   
                fragmentation_count = 0
                total_qrs = 0
   
                for i in range(min(len(q_peaks), len(s_peaks))):
                    if np.isnan(q_peaks[i]) or np.isnan(s_peaks[i]):
                        continue
           
                    qrs_start = int(q_peaks[i])
                    qrs_end = int(s_peaks[i])
       
                    if qrs_start < qrs_end and qrs_end < len(cleaned_signal):
                        # Count direction changes within the QRS complex
                        qrs_segment = cleaned_signal[qrs_start:qrs_end]
                        if len(qrs_segment) > 2:  # Need at least 3 points to detect changes
                            direction_changes = np.sum(np.diff(np.signbit(np.diff(qrs_segment))))
               
                            # More than 2 direction changes indicates fragmentation
                            if direction_changes > 2:
                                fragmentation_count += 1
                            total_qrs += 1
   
                if total_qrs > 0:
                    lead_features['QRS_Fragmentation_Ratio'] = fragmentation_count / total_qrs

                # print(lead_features)
    # print('done done')    
        # print( lead_features.keys() )
    except Exception as e:
        # Detailed exception handling can be uncommented for debugging
        # print(f"Error in waveform delineation for lead {lead_idx}: {e}")
        pass
   
    return lead_idx, lead_features

def process_single_record(record, desired_sampling_rate=100, train=True, feature_config=None):
    """Process a single ECG record and extract selected features."""
    # Load signal data
    signals, fields = load_signals(record)
    original_fs = fields["fs"]
   
    # Calculate expected length once
    expected_length = int(len(signals) * desired_sampling_rate / original_fs)
   
    # Setup - feature names will be determined by the feature_config
    n_leads = signals.shape[1] if len(signals.shape) > 1 else 1
   
    # # Create a test lead to get feature names
    # test_lead_idx, test_features = process_single_lead(
    #     signals[:, 0], 0, original_fs, desired_sampling_rate, expected_length, feature_config
    # )
    # feature_names = list(test_features.keys())
   
    # Reshape if needed
    if len(signals.shape) == 1:
        signals = signals.reshape(-1, 1)
   
    # Process each lead in parallel
    process_lead = partial(
        process_single_lead,
        original_fs=original_fs,
        desired_sampling_rate=desired_sampling_rate,
        expected_length=expected_length,
        feature_config=feature_config
    )
   
    # Determine number of cores to use
    num_cores = min(n_leads, multiprocessing.cpu_count())
   
   
   
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(
            process_lead,
            [(signals[:, i], i) for i in range(n_leads)]
        )

    feature_names = list( results[0][1].keys() )

   
    # text = " ".join(feature_names)
    # print(f"Feature names: {text}")
    # Save to a file
    # with open("featureOrder.txt", "w") as file:
    #     file.write(text)
   
    # Process leads in parallel
    features_list = [{name: np.nan for name in feature_names} for _ in range(n_leads)]

   

    # Reorganize results in correct lead order
    for lead_idx, lead_features in results:
        features_list[lead_idx] = lead_features
   
    # Convert to numpy array
    features_array = np.array([list(d.values()) for d in features_list])


    # print(f"Processed record: {record}, features shape: {features_array.shape}")
   
    # Imputation for missing features
    # if train:
    #     # Calculate median for each feature column, ignoring NaNs

    #     median_values = np.nanmedian(features_array, axis=0) # median per channel

    #     for iColumn in range( features_array.shape[1]):
    #         nan_idx = np.where( np.isnan( features_array[:, iColumn]  ) )[0]
    #         features_array[ nan_idx, iColumn] = median_values[iColumn]  #

        # medians_by_lead = []
        # for lead_idx in range(features_array.shape[0]):
        #     lead_medians = {}
        #     for j, feature_name in enumerate(feature_names):
        #         print( features_array[lead_idx, j] )
        #         lead_medians[feature_name] = np.nanmedian(features_array[lead_idx, j])
        #         if np.isnan(lead_medians[feature_name]):
        #             lead_medians[feature_name] = 0.0
        #     medians_by_lead.append(lead_medians)
   
        # # Impute missing values
        # for lead_idx in range(features_array.shape[0]):
        #     for j, feature_name in enumerate(feature_names):
        #         if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
        #             features_array[lead_idx, j] = medians_by_lead[lead_idx][feature_name]
   
        # feature_medians = medians_by_lead
   
    # elif feature_medians is not None:
    #     # Apply saved medians to test data
    #     for lead_idx in range(min(features_array.shape[0], len(feature_medians))):
    #         for j, feature_name in enumerate(feature_names):
    #             if j < len(feature_names) and feature_name in feature_medians[lead_idx]:
    #                 if np.isnan(features_array[lead_idx, j]) or np.isinf(features_array[lead_idx, j]):
    #                     features_array[lead_idx, j] = feature_medians[lead_idx][feature_name]
   
    # Final safety using vectorized operation
    # features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
   
    if train:
        label = load_label(record)
        label = np.array(label).astype(np.long)
        source = get_source(load_header(record))
        # print('label', label)
        # print(f"Processed record: {record}, features shape: {features_array.shape}")
        return record, features_array, label, source#, feature_medians
    else:
        return record, features_array
   

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


def load_and_process_signal_train(records,
                                  desired_sampling_rate=100,
                                  train=True,
                                  batch_size=5,
                                  feature_config=None):
    """
    Process multiple ECG records in parallel batches with configurable feature selection.
   
    Args:
        records: List of record paths or a single record path
        desired_sampling_rate: Target sampling rate (default: 100 Hz)
        train: Whether this is training data
        feature_medians: Dictionary of median values for imputation
        batch_size: Number of records to process in parallel (default: 5)
        feature_config: Dictionary of feature groups to enable/disable:
            {
                'basic_intervals': True,      # QRS duration, QT interval
                'amplitude': True,        # R-peak, P-wave amplitudes
                'hrv': True,                  # Heart rate variability metrics
                'p_wave': True,               # P wave features
                'st_segment': True,           # ST segment analysis
                't_wave': True,               # T wave morphology
                'pr_interval': True,          # PR interval analysis
                'qtc': True,                  # Corrected QT interval
                'qrs_fragmentation': True     # QRS fragmentation detection
            }
   
    Returns:
        If single record and train=True: features_array, label, source, feature_medians
        If single record and train=False: features_array
        If multiple records: list of processed record results
    """
    import time
   
    # If records is a single record path, convert to list
    if isinstance(records, str):
        single_record = True
        records = [records]
    else:
        single_record = False
   
    results = []
    total_start_time = time.time()
   
    # Process records in batches
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: records {i+1} to {min(i+batch_size, len(records))}")
   
        # Start timing the batch
        batch_start_time = time.time()
   
        # Use ProcessPoolExecutor to process records in parallel
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for record in batch_records:
                future = executor.submit(
                    process_single_record,
                    record=record,
                    desired_sampling_rate=desired_sampling_rate,
                    train=train,
                    feature_config=feature_config
                )
                futures.append(future)
       
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    record_result = future.result()
                    # print( 'record_result', record_result[-2])
       
                    results.append(record_result)
       
                except Exception as e:
                    print(f"Error processing record: {e}")
   
        # End timing the batch
        batch_end_time = time.time()
        # print the shape of the features for each record in the batch
   
        # if results:
        #     print(f"Batch {i//batch_size + 1} results: {len(results)} records processed")
        #     for j, result in enumerate(results[-len(batch_records):]):
        #         record_name, features_array = result[0], result[1]
        #         print(f"Record {record_name}: features shape {features_array.shape}")
        # print(f"Batch {i//batch_size + 1} completed: Processed {len(batch_records)} records in {batch_end_time - batch_start_time:.2f} seconds")
   
    # If processing a single record, return only its results
    if single_record and results:
        # For training mode
        if  not train and len(results[0]) == 2:
            _, features_array = results[0]
            return features_array
   
    total_end_time = time.time()
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds for {len(records)} records")
   
    return results



class CMLoss(nn.Module):
    """
    Clinical Metric (CM) Loss from Vicar et al. CinC2020-189.

    The true challenge metric is:
        CM = sum_{i,j} A_{ij} * W_{ij}
    where A is the (soft) confusion matrix and W is the provided weight matrix.

    This loss returns:
        L = -CM
    so that minimizing L maximizes the challenge metric.
    """

    def __init__(self, weight_matrix: torch.Tensor, eps: float = 1e-6):
        """
        Args:
            weight_matrix: FloatTensor of shape (c, c) containing w_{ij}.
            eps: Small constant to stabilize division.
        """
        super().__init__()
        self.W = weight_matrix
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: FloatTensor of shape (batch, c), raw model outputs.
            labels: FloatTensor of shape (batch, c), one-hot (or multi-hot) ground truth.

        Returns:
            loss: Scalar tensor = - sum_{i,j} A_{ij} * W_{ij}
        """
        # 1) Convert logits to probabilities
        p = torch.softmax( logits , dim=1)         # shape (b, c)
   
        L = labels.float()    # shape (b, c)    

        # # print( torch.sum(p,1) )
        # # print( p.shape, L.shape)

        # # 2) Continuous OR: B = L OR p = L + p - L*p
        # B = L + p - L * p                 # shape (b, c)

        # # print( B.shape)
        # # print(B)

        # # 3) Normalizer N: sum across classes then expand
        # N = B.sum(dim=1, keepdim=True)    # shape (b, 1)
        # N = N.clamp(min=self.eps).expand_as(B)  # shape (b, c)

        # # 4) Soft confusion matrix A = L^T @ (p / N)
        # R_over_N = p / N                  # shape (b, c)
        # A = L.t() @ R_over_N              # shape (c, c)

        N = L.sum(dim=0, keepdim=True)
        # print(N)
        N = N.clamp(min=self.eps).expand_as(L)
        # print(N.shape)
        # print(N)
        L_r = L / N
        A = L_r.t() @ p

        # print(A)

        # 5) Weighted sum with W
        cm_value = (A * self.W).sum()     # scalar

        # 6) Return negative for minimization
        return -cm_value

def optimizer_scheduler(optimizer, initial_lr, ep):
    if ep % 30 == 0:
        # Reduce LR
        expo = int(ep / 30 )
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * (0.1)**(expo)
   
    return optimizer







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
   

    scaler = StandardScaler()
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)
    print(f"Found {num_records} records")

#     # Get full paths for records
    record_paths = [os.path.join(data_folder, record) for record in records]

    # Process records in batches of 5 (optimal for EC2 g4dn.4xlarge with 16 vCPUs)
#     print("Processing records in parallel batches...")
    basic_config = {
    'basic_intervals': True,   # Keep QRS, QT intervals
    'amplitude': True,         # Keep R-peak amplitudes
    'hrv': True,
    'advanced_hrv': False,               # Keep heart rate variability metrics
    'p_wave': True,        # Keep P wave features
    'st_segment': False,       # Disable ST segment analysis
    't_wave': False,           # Disable T wave features
    'pr_interval': False,      # Disable PR interval features
    'qtc': False,              # Disable QTc features
    'qrs_fragmentation': False, # Disable QRS fragmentation detection
    'hrv_non_linear': False,  # Enable non-linear HRV features
    'hrv_welch': False,  # Enable Welch's method for HRV
     
}
   

    training_records = []
    print(f"Original number of records: {len(record_paths)}")

    # for i, path in enumerate(record_paths):
    #     source = load_source(path)
    #     label = load_label(path)
    #     print(source)
    #     # Skip negative samples from CODE-15
    #     if source == 'CODE-15%' and label == 0:
    #         pass
    #     else:
    #         training_records.append(path)

    # print(f"After filtering: {len(training_records)} records")

    # Filter records: keep all positive samples, only 1% of CODE-15 negative samples
    # training_records = []
    # code15_neg_records = []  # Store CODE-15 negative samples for sampling
    # print(f"Original number of records: {len(record_paths)}")

    # for i, path in enumerate(record_paths):
   
    #     source = load_source(path)
    #     label = load_label(path)

   
   
    #     if source == 'CODE-15%' and label == 0:
    #         code15_neg_records.append(path)
    #     else:
    #         training_records.append(path)

    # #camilo commneted this as there are not code 15 / jul 18
    # print('negative',len(code15_neg_records))
    # sample_size = max(1, int(0.01 * len(code15_neg_records)))  
   

    # # random.seed(42)  
    # sampled_code15_neg = random.sample(code15_neg_records, sample_size)

    # training_records.extend(code15_neg_records)

    # print(f"CODE-15 negative samples: {len(code15_neg_records)} total, {sample_size} sampled (1%)")
    # print(f"After filtering: {len(training_records)} records")
    # print(f"Sampling ratio for CODE-15 negatives: {sample_size}/{len(code15_neg_records)} = {100*sample_size/max(1,len(code15_neg_records)):.2f}%")


    calculate_features = True # True

    if calculate_features:
        results = load_and_process_signal_train(
            record_paths, #record_paths,
            desired_sampling_rate=100,
            train=True,
            batch_size=5,  # Process 5 records at a time,
            feature_config=basic_config
        )

        # Extract features, labels, and other data
        X_train_features = []
        y_train = []
        sources = []

        for record_path, features, label, source in results:
            # print(label)
            X_train_features.append(features)
            y_train.append(label)
            sources.append(source)


        X_train_features = np.array(X_train_features)
        y_train = np.array(y_train)


   
        # print(f"X_train_features shape: {X_train_features.shape}")  # Should be (num_samples, 12, 6)
        # print(f"y_train shape: {y_train.shape}")

        # # # # # ################################################################################################
        # # # # # # Optionally save/load features to avoid reprocessing
        # np.savez("fold_2_training_features_new_val_12_100Hz.npz", X_train_features=X_train_features, y_train=y_train, sources=sources)
   
    else:
        data = np.load("fold_2_training_features_new_val_12_100Hz.npz", allow_pickle=True)

        # feat_idx = [ i for i in range(len( feature_names) ) if feature_names[i] in desired_feat ]
        X_train_features = data['X_train_features']#[:, :, feat_idx]  # Ensure we only take the first 12 leads
        # print('ffff',X_train_features.shape)
   
        y_train = data['y_train']
   
        sources = data['sources']


   

    # feature_names = [
    #     'Mean_QRS_Duration_ms' ,
    #     'Std_QRS_Duration_ms' ,
    #     'Mean_QT_Interval_ms' ,
    #     'Std_QT_Interval_ms' ,
    #     'Mean_R_Amplitude' ,
    #     'Std_R_Amplitude' ,
    #     'QRS_Net_Deflection' ,
    #     'RR_Interval_RMSSD_ms',
    #     'HRV_SD1' ,
    #     'HRV_SD2' ,
    #     'HRV_SD1SD2' ,
    #     'HRV_LF' ,
    #     'HRV_HF' ,
    #     'HRV_LF_norm',
    #     'HRV_LF_HF',
    #     'Mean_P_Amplitude' ,
    #     'Std_P_Amplitude' ,
    #     'Mean_P_Duration_ms' ,
    #     'Std_P_Duration_ms' ,
    #     'ST_Deviation' ,
    #     'ST_Deviation_Std' ,
    #     'Mean_T_Amplitude' ,
    #     'T_Wave_Inversion_Ratio' ,
    #     'Mean_PR_Interval_ms' ,
    #     'Std_PR_Interval_ms' ,
    #     'QTc_Bazett_ms' ,
    #     'QTc_Fridericia_ms' ,
    #     'QRS_Fragmentation_Ratio' ,
    #     'RR_SDNN_ms' ,
    #     'pNN50']


    # # define here which features to use
    # desired_feat = [
    # 'Mean_P_Duration_ms',
    # 'HRV_SD2',
    # 'HRV_SD1',
    # 'HRV_LF',
    # 'RR_Interval_RMSSD_ms',
    # 'Std_P_Amplitude',
    # 'Std_P_Duration_ms',
    # 'ST_Deviation_Std',
    # 'HRV_LF_norm',
    # 'Mean_T_Amplitude',
    # 'QTc_Bazett_ms',
    # 'ST_Deviation',
    # 'Mean_P_Amplitude',
    # 'Std_QT_Interval_ms',
    # 'Mean_QT_Interval_ms',
    # 'Mean_R_Amplitude',
    # 'Mean_PR_Interval_ms',
    # 'pNN50',
    # 'T_Wave_Inversion_Ratio',
    # 'HRV_HF',
    # 'Std_R_Amplitude',
    # 'QRS_Net_Deflection',
    # 'Std_PR_Interval_ms',
    # 'RR_SDNN_msQRS_Fragmentation_Ratio',
    # 'Mean_QRS_Duration_ms',
    # 'Std_QRS_Duration_ms',
    # 'HRV_SD1SD2',
    # 'QTc_Fridericia_ms'  , 'HRV_LF_HF'   ]

    # desired_feat = desired_feat[:26]

   
   
    ################################################################################################
    # from imblearn.over_sampling import SMOTE

    # smote = SMOTE(
    #     sampling_strategy='auto',  # Balance all classes to match majority class
    #     random_state=42,
    #     k_neighbors=5,  # Number of nearest neighbors to use
    # )


    # Split data into train and validation sets with specific class distribution
    # First, separate data by class
    # class_0_indices = np.where(y_train == 0)[0]
    # class_1_indices = np.where(y_train == 1)[0]
   
    # # Determine validation set size (20% of total data)
    # val_size = int(.2 * len(y_train))
   
    # # Validation set: 95% class 0, 5% class 1
    # val_class_0_size = int(0.95 * val_size)
    # val_class_1_size = val_size - val_class_0_size
   
    # # Ensure we don't request more samples than available
    # val_class_0_size = min(val_class_0_size, len(class_0_indices))
    # val_class_1_size = min(val_class_1_size, len(class_1_indices))

    # # print( 'pos in validaiton', 100*val_class_1_size/(val_size) )
   
    # # Randomly select samples for validation
    # np.random.RandomState(42).shuffle(class_0_indices)
    # # np.random.shuffle(class_1_indices)
   
    # val_class_0_indices = class_0_indices[:val_class_0_size]

   
    # idx_eligible_pos = np.where( ( sources!='CODE-15%') &  ( y_train == 1 )  )[0]
    # np.random.RandomState(42).shuffle(idx_eligible_pos)
   


    # val_class_1_indices = idx_eligible_pos[:val_class_1_size]
   
    # # # Combine validation indices and the remaining for training
    # val_indices = np.concatenate([val_class_0_indices, val_class_1_indices])
    # train_indices = np.setdiff1d(np.arange(len(y_train)), val_indices)
   
    # # Create train and validation sets
    # X_train = X_train_features[train_indices]
    # y_train_split = y_train[train_indices]
 

    # X_val = X_train_features[val_indices]
    # y_val = y_train[val_indices]
   
    ############################################################################
    # Split data into train and validation sets with specific class distribution
    # First, separate data by class
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]
   
    # Determine validation set size (20% of total data)
    val_size = int(0.2 * len(y_train))
   
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
    ###########################################



    # print( 'pos in validaiton', 100*np.sum(y_val==1)/len(y_val) )

    ### Pre-process to handle infinite values
    print("Checking for and replacing infinite values...")
    # Replace inf/-inf with NaN so the imputer can handle them properly
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_val = np.nan_to_num(X_val, nan=np.nan, posinf=np.nan, neginf=np.nan)



    ### Lead-wise median imputation with saving imputers
    print("Applying lead-wise median imputation...")
    shape_original_train = X_train.shape
    shape_original_val = X_val.shape

    # Store imputers for each lead
    lead_imputers = []

    for lead_idx in range(shape_original_train[1]):  # For each of the 12 leads
        print(f"Processing lead {lead_idx + 1}/12...")
        
        # Extract data for this specific lead
        lead_train_data = X_train[:, lead_idx, :]  # Shape: (samples, features)
        lead_val_data = X_val[:, lead_idx, :]      # Shape: (samples, features)
        
        # Create and fit imputer for this lead
        lead_imputer = IterativeImputer( random_state=0, max_iter=10 ) #SimpleImputer(missing_values=np.nan, strategy='median')
        X_train[:, lead_idx, :] = lead_imputer.fit_transform(lead_train_data)
        X_val[:, lead_idx, :] = lead_imputer.transform(lead_val_data)
        
        # is there are nan values
        print(f"Lead {lead_idx + 1} - NaN values in training set: {np.isnan(X_train[:, lead_idx, :]).any()}")


        # Store the imputer for this lead
        lead_imputers.append(lead_imputer)

    # Store all lead imputers (you'll need this for prediction)
    imputer = lead_imputers  # Use this instead of the single imputer

   
    print(f"Training set: {len(y_train_split)} samples ({np.sum(y_train_split == 0)} class 0, {np.sum(y_train_split == 1)} class 1)")
    print(f"Validation set: {len(y_val)} samples ({np.sum(y_val == 0)} class 0, {np.sum(y_val == 1)} class 1)")

    print(f"X_train shape: {X_train.shape}")  # Should be (num_samples, 12, 6)
   
    # print for nans
    print(f"NaN values in training set: {np.isnan(X_train).any()}")
    print(f"NaN values in validation set: {np.isnan(X_val).any()}")

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
    # if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any():
    #     raise ValueError("NaN values found in scaled features. Check the input data and scaling process.")

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

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    total_steps = EPOCHS * len(train_dataloader)
    scheduler = OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=total_steps,
    pct_start=0.1,      # First 10% for warmup
    div_factor=25,      # Initial LR = max_lr/25
    final_div_factor=1e4  # Final LR = max_lr/10000
)
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
            ranking_loss = ranking_hinge_loss(scores, batch_labels, margin=2.0, num_pairs=10000)


            loss = f_loss + (1 * ranking_loss) 
     
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_labels.cpu().numpy())

        # Compute training metrics
        epoch_loss /= len(train_dataset)
        train_accuracy = (torch.tensor(all_train_predictions) == torch.tensor(all_train_labels)).sum().item() / len(all_train_labels)
        train_f1 = f1_score(all_train_labels, all_train_predictions)
        
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
        val_f1 = f1_score(val_ground_truth, val_predictions)
        
        # Calculate challenge score on validation set
        # Use fewer permutations during training for speed
        val_challenge_score = compute_challenge_score(
            val_ground_truth, 
            val_probabilities,
            num_permutations=10**4  # Reduced from 10^4 for faster computation during training
        )

        
        print(f'Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}, Challenge Score: {val_challenge_score:.4f}')
        
        # Save model if challenge score improves
        if val_challenge_score > best_challenge_score:
            best_challenge_score = val_challenge_score
            patience_counter = 0
            print(f"New best challenge score: {best_challenge_score:.4f}, saving model...")
            
            # Save the model
            os.makedirs(model_folder, exist_ok=True)    
            save_model(model_folder, model, scaler, imputer)
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
  
    if best_challenge_score < 0.317:
        print("Warning: Best validation challenge score is below 0.3, indicating poor model performance.")
        print(f"Best challenge score achieved: {best_challenge_score:.4f}")
        raise ValueError(f"Model training did not achieve a satisfactory challenge score {best_challenge_score}.")
    
    print(f"Training complete. Best validation challenge score: {best_challenge_score:.4f}")
    
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
        train=False)
    
    # Apply lead-wise imputation
    for lead_idx in range(features_array.shape[0]):  # For each lead
        lead_data = features_array[lead_idx, :].reshape(1, -1)  # Shape: (1, features)
        features_array[lead_idx, :] = model.lead_imputers[lead_idx].transform(lead_data)
    
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
    
    # Load lead-wise imputers (updated from feature_medians)
    lead_imputers_path = os.path.join(model_folder, 'lead_imputers.joblib')
    if os.path.exists(lead_imputers_path):
        lead_imputers = joblib.load(lead_imputers_path)
    else:
        # Fallback to old method for backward compatibility
        feature_medians_path = os.path.join(model_folder, 'feature_medians.pkl')
        with open(feature_medians_path, 'rb') as f:
            lead_imputers = pickle.load(f)
    
    # Create and load model - fix input_dim to match training (6, not 12)
    model = conv_model(input_dim=12, num_leads=12, num_classes=NUM_CLASSES).to(DEVICE)
    
    if model_path.endswith('.pth'):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Store preprocessing components with the model
    model.feature_scaler = feature_scaler
    model.lead_imputers = lead_imputers  # Updated to use lead_imputers
    
    if verbose:
        print('Model loaded successfully.')
        print(f'Loaded {len(lead_imputers)} lead-wise imputers')
    
    return model

def save_model(model_folder, model, scaler, imputer):
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
        pickle.dump(imputer, f)
    
    # 4. Also save a combined checkpoint for backward compatibility
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_medians': imputer
    }, os.path.join(model_folder, 'transformer_model.pt'))
    
    print(f"Model and components saved to {model_folder}")