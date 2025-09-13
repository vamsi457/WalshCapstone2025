# Libraries imports
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, welch, spectrogram
from numpy.fft import fft, ifft, fftfreq
from scipy.stats import skew, kurtosis, pearsonr
from math import gcd
import logging

# Plotly imports (for optional UI)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Layout

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
################### Common Data Definitions ######################
EEG_BANDS = {
    'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
    'Beta': (12, 30), 'Gamma': (30, 40)
}
DESIRED_FS = 128  # Target sampling rate

################### Data Handling & Conversion ######################
def string2array(s: str) -> np.ndarray:
    """Converts a comma-separated string to a numpy float array."""
    if not isinstance(s, str):
        return np.array([])
    return np.fromstring(s, dtype=float, sep=',')

def get_time_vector(signal, sampling_rate):
    """Generates a time vector for a given signal."""
    return np.arange(len(signal)) / sampling_rate

def mindbigdata_22_extract_data(data_file):
    # Define channel prefixes
    channel_prefixes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    # Build column labels for dask
    column_labels = ['label']
    for prefix in channel_prefixes:
        column_labels.extend([f'{prefix}-{i}' for i in range(256)])

    # Read CSV data from tmp_data using dask
    df = dd.read_csv(data_file, names=column_labels, header=0)

    def extract_channel_rows(row, event_idx):
        label = row['label']
        result = []
        for id_idx, prefix in enumerate(channel_prefixes):
            channel_cols = [f'{prefix}-{i}' for i in range(256)]
            data_str = ','.join(str(row[col]) for col in channel_cols)
            result.append({'id': id_idx, 'event': event_idx, 'device': 'EP', 'channel': prefix, 'code': int(label), 'size': 256, 'data': data_str})
        return result

    def process_partition(partition):
        rows = []
        for event_idx, (_, row) in enumerate(partition.iterrows()):
            rows.extend(extract_channel_rows(row, event_idx))
        return dd.from_pandas(pd.DataFrame(rows), npartitions=1)

    new_ddf = df.map_partitions(process_partition)
    # fill id column value with index number
    new_ddf['id'] = new_ddf.index

    return new_ddf

################# Signal Processing Atomic Functions ####################
def resample_waveform(data: np.ndarray, target_fs: float, original_fs: float) -> np.ndarray:
    if original_fs == target_fs:
        return data
    common_divisor = gcd(int(target_fs), int(original_fs))
    return resample_poly(data, int(target_fs / common_divisor), int(original_fs / common_divisor))

def adjust_signal_length(signal: np.ndarray, sampling_rate: float, target_duration_s: float, criteria: str) -> np.ndarray:
    target_samples = int(target_duration_s * sampling_rate)
    if len(signal) > target_samples and criteria:
        windows = np.lib.stride_tricks.sliding_window_view(signal, window_shape=target_samples)
        if criteria == 'min_abs_amplitude':
            best_idx = np.max(np.abs(windows), axis=1).argmin()
        elif criteria == 'min_variance':
            best_idx = np.var(windows, axis=1).argmin()
        else:
            best_idx = 0
        signal = windows[best_idx]
    
    if len(signal) >= target_samples:
        return signal[:target_samples]
    return np.pad(signal, (0, target_samples - len(signal)), 'constant')

def band_pass_filter(data: np.ndarray, low_cut: float, high_cut: float, fs: float, N: int = 5) -> np.ndarray:
    """Applies a Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(N, [low, high], btype='bandpass')
    return filtfilt(b, a, data)

def apply_frequency_filters(signal: np.ndarray, fs: float, params: dict) -> np.ndarray:
    nyquist = 0.5 * fs
    # Band-pass filter
    low, high = params['low_cut_bp'] / nyquist, params['high_cut_bp'] / nyquist
    b, a = butter(params['filter_order_bp'], [low, high], btype='bandpass')
    filtered_signal = filtfilt(b, a, signal)
    # Notch filter
    notch_freq = params['notch_freq']
    if 0 < notch_freq < nyquist:
        b, a = iirnotch(notch_freq / nyquist, params['q_factor_notch'])
        filtered_signal = filtfilt(b, a, filtered_signal)
    return filtered_signal

def normalize_amplitude(signal: np.ndarray, method: str) -> np.ndarray:
    if method == 'z-score':
        std = np.std(signal)
        return (signal - np.mean(signal)) / std if std > 0 else np.zeros_like(signal)
    elif method == 'min-max':
        min_val, max_val = np.min(signal), np.max(signal)
        return (signal - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(signal)
    raise ValueError(f"Unknown normalization method: {method}")

def common_average_reference(signals: np.ndarray) -> np.ndarray:
    if signals.ndim != 2 or signals.shape[0] < 2:
        return signals
    return signals - np.mean(signals, axis=0)

################### Feature Extraction #####################

def zero_crossing_rate(signal: np.ndarray) -> float:
    """Compute the zero-crossing rate of a 1D signal array."""
    if signal.size < 2:
        return 0.0
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    return len(zero_crossings) / (len(signal) - 1)

def compute_psd(data: np.ndarray, fs: float, nperseg_factor: float = 2.0, noverlap_factor: float = 1.0) -> (np.ndarray, np.ndarray):
    """
    Computes Power Spectral Density (PSD) using Welch's method.
    Adjusts nperseg and noverlap based on signal length to avoid errors.
    """
    nperseg_val = int(nperseg_factor * fs)
    noverlap_val = int(noverlap_factor * fs)

    if nperseg_val > len(data):
        nperseg_val = len(data)
        noverlap_val = nperseg_val // 2 if nperseg_val // 2 >= 1 else 0
    
    if nperseg_val == 0: # Handle very short signals
        return np.array([]), np.array([])

    freq, psd = welch(data, fs, nperseg=nperseg_val, noverlap=noverlap_val)
    return freq, psd

def extract_eeg_features1(signal: np.ndarray, sampling_rate: float, duration: float) -> dict:
    eeg_features = {}
    if signal.size == 0:
        return {}
        
    # Time-Domain (Full Signal)
    eeg_features['full_Mean'] = np.mean(signal)
    eeg_features['full_Std'] = np.std(signal)
    eeg_features['full_Var'] = np.var(signal)
    eeg_features['full_Skewness'] = skew(signal)
    eeg_features['full_Kurtosis'] = kurtosis(signal)
    eeg_features['full_PeakToPeak'] = np.ptp(signal)
    eeg_features['full_ZeroCrossingRate'] = zero_crossing_rate(signal)
    
    # Frequency-Domain (from PSD)
    nperseg = min(len(signal), int(2 * sampling_rate))
    if nperseg == 0: return eeg_features
    freqs, psd = welch(signal, sampling_rate, nperseg=nperseg)
    total_power = np.sum(psd)

    # Features for each band
    for band, (f_low, f_high) in EEG_BANDS.items():
        band_mask = (freqs >= f_low) & (freqs < f_high)
        if np.any(band_mask):
            band_power = np.sum(psd[band_mask])
            eeg_features[f'{band}_AbsolutePower'] = band_power
            eeg_features[f'{band}_RelativePower'] = band_power / total_power if total_power > 0 else 0
            
        else: 
            eeg_features[f'{band}_AbsolutePower'] = 0.0
            eeg_features[f'{band}_RelativePower'] = 0.0
            
    return eeg_features

def extract_eeg_features(signal: np.ndarray, sampling_rate: float, duration: float) -> (dict, np.ndarray, np.ndarray, dict):
    """
    Extracts various time-domain and frequency-domain features from a cleaned (and normalized) EEG signal.
    Also extracts features for all brain frequency bands (Delta, Theta, Alpha, Beta, Gamma).
    """
    eeg_features = {}

    # Time-Domain Statistical Features (full signal)
    eeg_features['full_Mean'] = np.mean(signal)
    eeg_features['full_StandardDeviation'] = np.std(signal)
    eeg_features['full_Variance'] = np.var(signal)
    eeg_features['full_Skewness'] = skew(signal)
    eeg_features['full_Kurtosis'] = kurtosis(signal)
    eeg_features['full_PeakToPeakAmplitude'] = np.max(signal) - np.min(signal)

    # Zero-Crossing Rate (ZCR)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    eeg_features['full_ZeroCrossingRate'] = len(zero_crossings) / duration if duration > 0 else 0

    # Hjorth Parameters
    eeg_features['full_Hjorth_Activity'] = np.var(signal)
    first_derivative = np.diff(signal)
    std_signal = np.std(signal)
    std_first_derivative = np.std(first_derivative)
    eeg_features['full_Hjorth_Mobility'] = std_first_derivative / std_signal if std_signal != 0 else 0

    second_derivative = np.diff(first_derivative)
    std_second_derivative = np.std(second_derivative)
    mobility_first_derivative = std_second_derivative / std_first_derivative if std_first_derivative != 0 else 0
    eeg_features['full_Hjorth_Complexity'] = mobility_first_derivative / eeg_features['full_Hjorth_Mobility'] if eeg_features['full_Hjorth_Mobility'] != 0 else 0

    # Frequency-Domain Features (from PSD)
    freq_spectrum, psd_spectrum = compute_psd(signal, sampling_rate)
    total_power = np.sum(psd_spectrum)

    # Extract features for each band (absolute/relative power, peak freq, time-domain stats)
    for band, (f_low, f_high) in EEG_BANDS.items():
        idx_band = np.where((freq_spectrum >= f_low) & (freq_spectrum < f_high))
        if idx_band[0].size > 0:
            band_power = np.sum(psd_spectrum[idx_band])
            eeg_features[f'{band}_AbsolutePower'] = band_power
            eeg_features[f'{band}_RelativePower'] = (band_power / total_power) if total_power > 0 else 0
            peak_freq_idx = idx_band[0][np.argmax(psd_spectrum[idx_band])]
            eeg_features[f'{band}_PeakFrequency'] = freq_spectrum[peak_freq_idx]
            # Time-domain features for band-passed signal
            band_signal = band_pass_filter(signal, f_low, f_high, sampling_rate)
            eeg_features[f'{band}_Mean'] = np.mean(band_signal)
            eeg_features[f'{band}_Std'] = np.std(band_signal)
            eeg_features[f'{band}_Var'] = np.var(band_signal)
            eeg_features[f'{band}_Skew'] = skew(band_signal)
            eeg_features[f'{band}_Kurtosis'] = kurtosis(band_signal)
            eeg_features[f'{band}_PeakToPeak'] = np.max(band_signal) - np.min(band_signal)
            # Zero-Crossing Rate (ZCR) for band
            band_zero_crossings = np.where(np.diff(np.sign(band_signal)))[0]
            eeg_features[f'{band}_ZeroCrossingRate'] = len(band_zero_crossings) / duration if duration > 0 else 0
            # Hjorth Parameters for band
            band_var = np.var(band_signal)
            band_first_derivative = np.diff(band_signal)
            band_std = np.std(band_signal)
            band_std_first_derivative = np.std(band_first_derivative)
            band_mobility = band_std_first_derivative / band_std if band_std != 0 else 0
            band_second_derivative = np.diff(band_first_derivative)
            band_std_second_derivative = np.std(band_second_derivative)
            band_mobility_first_derivative = band_std_second_derivative / band_std_first_derivative if band_std_first_derivative != 0 else 0
            band_complexity = band_mobility_first_derivative / band_mobility if band_mobility != 0 else 0
            eeg_features[f'{band}_Hjorth_Activity'] = band_var
            eeg_features[f'{band}_Hjorth_Mobility'] = band_mobility
            eeg_features[f'{band}_Hjorth_Complexity'] = band_complexity
        else:
            eeg_features[f'{band}_AbsolutePower'] = 0.0
            eeg_features[f'{band}_RelativePower'] = 0.0
            eeg_features[f'{band}_PeakFrequency'] = np.nan
            eeg_features[f'{band}_Mean'] = np.nan
            eeg_features[f'{band}_Std'] = np.nan
            eeg_features[f'{band}_Var'] = np.nan
            eeg_features[f'{band}_Skew'] = np.nan
            eeg_features[f'{band}_Kurtosis'] = np.nan
            eeg_features[f'{band}_PeakToPeak'] = np.nan
            eeg_features[f'{band}_ZeroCrossingRate'] = np.nan
            eeg_features[f'{band}_Hjorth_Activity'] = np.nan
            eeg_features[f'{band}_Hjorth_Mobility'] = np.nan
            eeg_features[f'{band}_Hjorth_Complexity'] = np.nan

    # Power Ratios
    theta_power = eeg_features.get('Theta_AbsolutePower', 0)
    alpha_power = eeg_features.get('Alpha_AbsolutePower', 0)
    beta_power = eeg_features.get('Beta_AbsolutePower', 0)
    eeg_features['full_Theta_Alpha_Ratio'] = theta_power / alpha_power if alpha_power != 0 else np.nan
    eeg_features['full_Alpha_Beta_Ratio'] = alpha_power / beta_power if beta_power != 0 else np.nan

    return eeg_features, freq_spectrum, psd_spectrum, EEG_BANDS


################### Pipeline Stage Functions ######################

def process_raw_signals(df_raw: pd.DataFrame, original_fs: float, params: dict) -> pd.DataFrame:
    """Stage 1: Takes raw data and applies signal processing."""
    processed_rows = []
    
    for _, row in df_raw.iterrows():
        new_row = row.to_dict()
        signal = string2array(row['data'])
        
        if signal.size > 0:
            resampled = resample_waveform(signal, params['target_sr'], original_fs)
            standardized = adjust_signal_length(resampled, params['target_sr'], params['target_duration_s'], params['segment_selection_criteria'])
            filtered = apply_frequency_filters(standardized, params['target_sr'], params)
            normalized = normalize_amplitude(filtered, params['normalization_method'])
            new_row['processed_signal'] = normalized.tolist()
        else:
            new_row['processed_signal'] = []
            
        processed_rows.append(new_row)
        
    return pd.DataFrame(processed_rows).drop(columns=['data', 'size'])

def _apply_car_and_filter_to_group(event_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Helper to apply CAR and correlation filtering to a single event group."""
    signals_list = [s for s in event_df['denoised_signal'].tolist() if s]
    if len(signals_list) < 2:
        event_df['final_signal_for_features'] = event_df['denoised_signal']
        return event_df

    signals_array = np.array(signals_list)
    car_signals = common_average_reference(signals_array)
    mean_car_signal = np.mean(car_signals, axis=0)

    correlations = []
    filtered_indices, filtered_signals = [], []
    for i, channel_signal in enumerate(car_signals):
        if np.std(channel_signal) > 0 and np.std(mean_car_signal) > 0:
            corr, _ = pearsonr(channel_signal, mean_car_signal)
            correlations.append(corr)
            if corr >= params['correlation_threshold']:
                filtered_indices.append(i)
                filtered_signals.append(channel_signal.tolist())
        else:
            correlations.append(np.nan)

    if correlations:
        event_id = event_df['event'].iloc[0]
        logging.info(f"Event {event_id}: Correlation Min: {np.nanmin(correlations):.4f}, Max: {np.nanmax(correlations):.4f}")

    if not filtered_indices:
        # Return empty DataFrame with same columns as event_df
        return pd.DataFrame(columns=event_df.columns)

    result_df = event_df.iloc[filtered_indices].copy()
    result_df['final_signal_for_features'] = filtered_signals
    return result_df

def extract_features_from_processed(df_processed: pd.DataFrame, params: dict, is_multichannel: bool, no_car: bool) -> pd.DataFrame:
    """Stage 2: Applies noise reduction, CAR (if multi-channel), and extracts features."""
    noise_df = df_processed[df_processed['code'] == -1]
    signal_df = df_processed[df_processed['code'] != -1].copy()

    valid_noise_signals = [s for s in noise_df['processed_signal'].tolist() if hasattr(s, '__len__') and len(s) > 0]
    if valid_noise_signals:
        logging.info(f"Applying noise reduction using {len(valid_noise_signals)} signals.")
        mean_noise_profile = np.mean(np.array(valid_noise_signals), axis=0)
        signal_df['denoised_signal'] = signal_df['processed_signal'].apply(
            lambda s: (np.array(s) - mean_noise_profile).tolist() if hasattr(s, '__len__') and len(s) > 0 else []
        )
    else:
        logging.warning("No valid noise signals found. Skipping noise reduction.")
        signal_df['denoised_signal'] = signal_df['processed_signal']

    if is_multichannel and not no_car:
        logging.info("Applying CAR and correlation filtering.")
        final_signals_df = signal_df.groupby('event', group_keys=False).apply(_apply_car_and_filter_to_group, params)
    else:
        final_signals_df = signal_df.copy()
        final_signals_df['final_signal_for_features'] = final_signals_df['denoised_signal']

    if final_signals_df.empty:
        logging.warning("DataFrame is empty after processing/filtering. No features will be extracted.")
        return pd.DataFrame()

    logging.info("Extracting features from final signals...")
    features_list = []
    for _, row in final_signals_df.iterrows():
        signal_list = row['final_signal_for_features']
        if signal_list:
            signal_array = np.array(signal_list)
            raw_features, freq_spectrum, psd_spectrum, band_Details = extract_eeg_features(signal_array, params['target_sr'], params['target_duration_s'])
            # If extract_eeg_features returns a tuple, use only the first element (the features dict)
            # if isinstance(raw_features, tuple):
            #     raw_features = raw_features[0]
            prefixed_features = {f'feature_{k}': v for k, v in raw_features.items()}
            # freq_spectrum_feature = {"freq_spectrum": freq_spectrum, "psd_spectrum": psd_spectrum}
            prefixed_features["freq_spectrum"] = freq_spectrum
            prefixed_features["psd_spectrum"] = psd_spectrum
            features_list.append(prefixed_features)
            # features_list.append(freq_spectrum_feature)
        else:
            features_list.append({})
    features_df = pd.DataFrame(features_list, index=final_signals_df.index)
    
    # Combine metadata with new features
    return pd.concat([final_signals_df.drop(columns=['processed_signal', 'denoised_signal', 'final_signal_for_features']), features_df], axis=1)


def aggregate_multichannel_features(features_df: pd.DataFrame, is_multichannel: bool) -> pd.DataFrame:
    """Stage 3: Aggregates features across channels for each event if multi-channel."""
    if not is_multichannel:
        logging.info("Skipping aggregation for single-channel device.")
        return features_df
    
    feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
    if not feature_cols: return features_df
        
    grouping_keys = ['event', 'device', 'code']
    aggregated_df = features_df.groupby(grouping_keys)[feature_cols].mean().reset_index()
    logging.info(f"Aggregated {len(features_df)} channel entries into {len(aggregated_df)} event entries.")
    return aggregated_df

def select_best_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Stage 4: Selects the top k features from an aggregated/feature DataFrame."""
    if df.empty: return pd.DataFrame()
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if not feature_cols: return df

    X = df[feature_cols].copy().fillna(0) # Simple imputation for selection
    y = df['code'].copy()
    
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    logging.info(f"Selected {len(selected_features)} best features.")
    
    # Keep essential columns for context
    id_cols = [col for col in ['event', 'device', 'code'] if col in df.columns]
    return df[id_cols + selected_features]

################### Machine Learning Functions ###################
def prepare_signals_for_training(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'processed_signal' column (time-series data) into a feature matrix.
    Each time point in the signal becomes a separate feature.
    """
    if 'processed_signal' not in processed_df.columns or 'code' not in processed_df.columns:
        logging.error("Input DataFrame must contain 'processed_signal' and 'code' columns.")
        return pd.DataFrame()

    # Removing Noise signals
    processed_df = processed_df[processed_df["code"] != -1]

    # More robust filter that works for both lists and ndarrays.
    df = processed_df[processed_df['processed_signal'].apply(lambda x: hasattr(x, '__len__') and len(x) > 0)].copy()

    if df.empty:
        logging.warning("No valid signals found in the processed DataFrame to prepare for training.")
        return pd.DataFrame()

    # Use np.vstack for robust creation of the feature matrix from lists or arrays.
    signal_matrix = np.vstack(df['processed_signal'].values)
    
    # Create new column names for each time point (feature)
    feature_names = [f'feature_timepoint_{i}' for i in range(signal_matrix.shape[1])]
    
    # Create a new DataFrame for the features
    features_df = pd.DataFrame(signal_matrix, columns=feature_names, index=df.index)
    
    # Combine the new features DataFrame with the essential 'code' column from the original df
    model_ready_df = pd.concat([df[['code']], features_df], axis=1)
    
    return model_ready_df

def prepare_for_training(df, label_col='code', test_size=0.2, random_state=42):
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    if not feature_cols:
        logging.error("No feature columns (starting with 'feature_') found in the DataFrame for training.")
        return None, np.array([]), np.array([]), np.array([]), np.array([])

    X = df[feature_cols].copy().fillna(0)
    y = df[label_col].copy()
    
    if X.empty or y.empty:
        logging.error("Feature matrix (X) or label vector (y) is empty.")
        return None, np.array([]), np.array([]), np.array([]), np.array([])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if len(np.unique(y)) > 1:
        return scaler, *train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        logging.warning("Only one class present in data. Cannot perform stratified split.")
        return scaler, *train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name='Model'):
    if X_train.size == 0 or y_train.size == 0:
        print(f"--- {model_name} ---\nSkipping training: No data available.")
        return model, 0, ""
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"--- {model_name} ---\nAccuracy: {acc:.4f}\nClassification Report:\n{report}")
    return model


def train_and_evaluate_cnn(X_train, X_test, y_train, y_test, num_classes):
    """Builds, trains, and evaluates a 1D CNN model."""
    if X_train.size == 0:
        print("--- CNN ---\nSkipping training: No data available.")
        return None, 0, ""
        
    # Reshape data for Conv1D: (samples, timesteps, features/channels)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # One-hot encode labels
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("--- CNN ---\nTraining model...")
    model.fit(X_train_cnn, y_train_cat, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    
    loss, acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    y_pred_prob = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"Accuracy: {acc:.4f}\nClassification Report:\n{report}")
    return model

################## Optional Plotly UI for Jupyter ###################
def preprocess_signal_for_viz(signal: np.ndarray, original_sr: float):
    params = {'target_sr': 128, 'target_duration_s': 2, 'low_cut_bp': 0.5, 'high_cut_bp': 40, 'notch_freq': 50, 'filter_order_bp': 5, 'q_factor_notch': 30.0, 'segment_selection_criteria': 'min_abs_amplitude', 'normalization_method': 'z-score'}
    resampled = resample_waveform(signal, params['target_sr'], original_sr)
    standardized = adjust_signal_length(resampled, params['target_sr'], params['target_duration_s'], params['segment_selection_criteria'])
    filtered = apply_frequency_filters(standardized, params['target_sr'], params)
    return normalize_amplitude(filtered, params['normalization_method'])

def _add_traces_to_fig(fig, data, fs, selected_segments, compute_func, **kwargs):
    if "Original" in selected_segments:
        x, y = compute_func(data, fs, **kwargs)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Original'))
        
    if any(s != "Original" for s in selected_segments):
        processed_data = preprocess_signal_for_viz(data, fs)
        if "Processed" in selected_segments:
            x, y = compute_func(processed_data, DESIRED_FS, **kwargs)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Processed'))
        
        segments_dict = {band: butter(5, [f_low / (0.5*DESIRED_FS), f_high / (0.5*DESIRED_FS)], btype='bandpass', output='sos') for band, (f_low, f_high) in EEG_BANDS.items()}
        for band, sos in segments_dict.items():
            if f"{band} Band" in selected_segments:
                segment = filtfilt(sos[0], sos[1], processed_data)
                x, y = compute_func(segment, DESIRED_FS, **kwargs)
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{band} Band'))

def display_waveform(data, fs, title='Waveform', selected_segments=("Original",)):
    fig = go.Figure()
    _add_traces_to_fig(fig, data, fs, selected_segments, lambda d, s: (get_time_vector(d, s), d))
    fig.update_layout(title=title, xaxis_title='Time (s)', yaxis_title='Amplitude').show()

def display_fft(data, fs, title='FFT', selected_segments=("Original",)):
    fig = go.Figure()
    _add_traces_to_fig(fig, data, fs, selected_segments, lambda d, s: (fftfreq(len(d), 1/s), np.abs(fft(d))))
    fig.update_layout(title=title, xaxis_title='Frequency (Hz)', yaxis_title='Magnitude').show()

def display_psd(data, fs, title='PSD', selected_segments=("Original",)):
    fig = go.Figure()
    _add_traces_to_fig(fig, data, fs, selected_segments, lambda d, s: welch(d, s, nperseg=min(len(d), 256)))
    fig.update_layout(title=title, xaxis_title='Frequency (Hz)', yaxis_title='Power/Frequency').show()

class EEGDataFrameUI:
    def __init__(self, df_options, string2array_func, sampling_rate_map, display_waveform_func, display_fft_func, display_psd_func):
        self.df_options = df_options
        self.string2array = string2array_func
        self.sampling_rate = sampling_rate_map
        self.display_waveform = display_waveform_func
        self.display_fft = display_fft_func
        self.display_psd = display_psd_func
        
        self.df_selector = widgets.Dropdown(options=list(df_options.keys()), description='DataFrame:')
        self.index_selector = widgets.IntSlider(min=0, max=1, description='Index:', continuous_update=False, layout={'width': '50%'})
        self.graph_selector = widgets.Dropdown(options=["Waveform", "FFT", "PSD"], description='Graph:')
        
        segment_options = [("Original", "Original"), ("Processed", "Processed")] + [
            (f"{band} Band ({f_low}-{f_high} Hz)", f"{band} Band")
            for band, (f_low, f_high) in EEG_BANDS.items()
        ]
        self.segment_selector = widgets.SelectMultiple(options=segment_options, value=["Original"], description='Segments:')
        
        self.chk_target = widgets.Checkbox(value=False, description='Filter by Target')
        self.trg_selector = widgets.Dropdown(options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], description='Target:', disabled=True)

        self.info_label = widgets.Label()
        self.output = widgets.Output()

        self._setup_observers()
        self._on_df_change()
        self._display_ui()
    
    def _setup_observers(self):
        self.df_selector.observe(self._on_df_change, names='value')
        self.chk_target.observe(self._on_df_change, names='value')
        self.trg_selector.observe(self._on_df_change, names='value')
        for w in [self.index_selector, self.graph_selector, self.segment_selector]:
            w.observe(self._update_data, names='value')

    def _on_df_change(self, change=None):
        self.trg_selector.disabled = not self.chk_target.value
        df = self.df_options[self.df_selector.value]
        if self.chk_target.value:
            df = df[df['code'] == self.trg_selector.value]
        self.index_selector.max = max(0, len(df) - 1)
        self.index_selector.value = 0
        self._update_data()

    def _update_data(self, change=None):
        df = self.df_options[self.df_selector.value]
        if self.chk_target.value:
            df = df[df['code'] == self.trg_selector.value].reset_index(drop=True)
        
        with self.output:
            clear_output(wait=True)
            if df.empty or self.index_selector.value >= len(df):
                print("No data.")
                return
            
            row = df.iloc[self.index_selector.value]
            self.info_label.value = f"Device: {row.get('device', 'N/A')}, Channel: {row.get('channel', 'N/A')}, Target: {row.get('code', 'N/A')}"
            data = self.string2array(row["data"])
            fs = self.sampling_rate.get(self.df_selector.value, 128)
            
            graph_func = {'Waveform': self.display_waveform, 'FFT': self.display_fft, 'PSD': self.display_psd}.get(self.graph_selector.value)
            if graph_func:
                graph_func(data, fs, selected_segments=self.segment_selector.value)
            
    def _display_ui(self):
        controls = widgets.VBox([
            widgets.HBox([self.df_selector, self.graph_selector]),
            self.segment_selector,
            widgets.HBox([self.chk_target, self.trg_selector]),
            self.index_selector, 
            self.info_label
        ])
        display(widgets.VBox([controls, self.output]))

