import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import math
import warnings
import datetime
from numpy.linalg import lstsq
import os  # Import the os module

# =============================================================================
# 1. COMMON DATA PREPROCESSING & CLEANING (for the entire dataset)
# =============================================================================

# Load Data
file_path = "/mount/nas/disk02/Data/Health/Mental_Health/SAFER/20241212/clean/all_concat_2.csv"
data = pd.read_csv(file_path)
print("Initial Data Head:")
print(data.head())

# Fill missing values and drop irrelevant columns
data = data.fillna(0)
drop_columns = [
    'MED_AP', 'MED_AD', 'MED_MS', 'MED_SH', 'MED_czp', 'BPRS_AFF', 'BPRS_POS',
    'BPRS_NEG', 'BPRS_RES', 'BPRS_sum', 'YMRS_sum', 'MADRS_sum', 'HAMA_sum',
    'CS_status_1_score_cal', 'CS_status_2_bhv_score_cal', 'CS_status_2_NSSI',
    'marri', 'edu', 'occu', 'relig', 'insurance', 'ho_inc_month', 'convicted',
    'PH_no_physical', 'PH_tx_status', 'PH_tx_hospitalization', 'MED_onmed',
    'MED_duration', 'MED_compliance', 'MINI_MDx_text', 'DIG_text',
    'MINI_curr_epi', 'DIG_cat', 'VH_selfharm', 'VH_selfharm_count', 'crime',
    'CS_life_1_score_cal', 'CS_life_2_NSSI', 'CS_life_2_bhv_score_cal',
    'HCR_P_H_sum', 'HCR_P_C_sum', 'HCR_P_R_sum', 'HCR_P_sum', 'CTQ_EA',
    'CTQ_PA', 'CTQ_SA', 'CTQ_EN', 'CTQ_PN', 'CTQ_MD', 'BIS_M', 'BIS_NP',
    'BIS_sum', 'ASRS_IA', 'ASRS_HM', 'ASRS_HV', 'ASRS_sum', 'AUDIT_sum',
    'VE_tx_secl_endtime', 'VE_tx_rest_startime', 'VE_tx_rest_endtime','VE_tx_secl_startime'
]
drop_columns = [col for col in drop_columns if col in data.columns]
data.drop(columns=drop_columns, inplace=True)

# Feature Identification: Determine which columns are categorical or continuous.
excluded_columns = ['key', 'targetTime']
threshold = 20  # if unique values <= threshold, treat as categorical

categorical_columns = [
    col for col in data.columns
    if col not in excluded_columns and
        (data[col].nunique() <= threshold or data[col].dtype == 'object')
]

continuous_columns = [
    col for col in data.columns
    if col not in excluded_columns and
        col not in categorical_columns and
        data[col].dtype in ['int64', 'float64']
]

print("Categorical Columns:")
print(categorical_columns)
print("\nContinuous Columns:")
print(continuous_columns)

data.rename(columns={'이름': 'key'}, inplace=True)
# Recompute feature lists to reflect the new column name:
excluded_columns = ['key', 'targetTime']
categorical_columns = [
    col for col in data.columns
    if col not in excluded_columns and
        (data[col].nunique() <= threshold or data[col].dtype == 'object')
]
continuous_columns = [
    col for col in data.columns
    if col not in excluded_columns and
        col not in categorical_columns and
        data[col].dtype in ['int64', 'float64']
]
# =============================================================================
# 2. SPLIT DATA BY HOSPITAL (All remaining steps will be processed hospital-level)
# =============================================================================

def extract_hospital_name(idx_info):
    """Extract hospital name from the idxInfo column (first part before '-')"""
    return idx_info.split("-")[0]

# Add a new column for hospital using idxInfo
data['hospital'] = data['idxInfo'].apply(extract_hospital_name)

# Get the unique hospital names
hospitals = data['hospital'].unique()
print("Unique Hospitals:", hospitals)

# =============================================================================
# 3. DEFINE FUNCTIONS FOR SUBSEQUENT PROCESSING
# =============================================================================

# ----- Data Augmentation -----
class DataAugmentor:
    def __init__(self, sigma=0.5, num_augments=3): # Default num_augments is 5 now
        self.sigma = sigma
        self.num_augments = num_augments

    def time_warping(self, data):
        warping_factor = np.random.normal(1, self.sigma, size=data.shape)
        return data * warping_factor

    def augment_patient_data(self, df, target_columns, categorical_columns):
        positive_data = df[df['dangerous_action'] == 1]
        augmented_data = []
        patient_id = max(
            [int(id.split('_')[-1]) for id in df['key'].unique() if '_' in id] + [0]
        ) + 1

        for name, group in positive_data.groupby("key"):
            for _ in range(self.num_augments):
                new_patient_data = group.copy()
                new_patient_data["key"] = f"NewPatient_{patient_id}"
                patient_id += 1

                for col in target_columns:
                    new_patient_data[col] = self.time_warping(new_patient_data[col].to_numpy())

                for col in group.columns:
                    if col in categorical_columns:
                        continue
                    if pd.api.types.is_numeric_dtype(group[col]) and col not in target_columns:
                        new_patient_data[col] += np.random.normal(0, self.sigma, size=new_patient_data[col].shape)

                augmented_data.append(new_patient_data)

        augmented_df = pd.concat(augmented_data, ignore_index=True)
        return pd.concat([df, augmented_df], ignore_index=True)

# ----- Sensor Data Processing (Cosinor Analysis) -----
class CosinorAnalyzer:
    def __init__(self, data: pd.DataFrame, window_length_hours: float,
                 coverage_threshold: float = 0.5, sensor: str = 'ENMO',
                 name_col: str = '이름', time_col: str = 'targetTime'):
        self.data = data.copy()
        self.window_length_hours = window_length_hours
        self.coverage_threshold = coverage_threshold
        self.period = 24.0  # fixed period
        self.sensor = sensor
        self.name_col = name_col
        self.time_col = time_col
        self.sensor_col = f"{self.sensor}_mean"

        required_cols = [self.time_col, self.sensor_col, 'nonwearing', self.name_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain '{col}' column.")

        if not pd.api.types.is_datetime64_any_dtype(self.data[self.time_col]):
            raise ValueError(f"data['{self.time_col}'] must be a datetime type.")

        self.data.sort_values([self.name_col, self.time_col], inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        time_diffs = self.data.groupby(self.name_col)[self.time_col].diff().dropna()
        if len(time_diffs) == 0:
            raise ValueError("Not enough data points to determine sampling rate.")
        median_diff = time_diffs.median().total_seconds() / 60.0
        self.sampling_interval_minutes = median_diff

    def _prepare_window_data(self, df_group, ref_time: pd.Timestamp, global_reference_time):
        start_time = ref_time - pd.Timedelta(hours=self.window_length_hours)
        end_time = ref_time

        df_window = df_group[(df_group[self.time_col] >= start_time) &
                                        (df_group[self.time_col] <= end_time)]
        df_window = df_window[df_window['nonwearing'] == False]

        expected_points = int((self.window_length_hours * 60) / self.sampling_interval_minutes)
        actual_points = len(df_window)

        if actual_points == 0 or (actual_points / expected_points) < self.coverage_threshold:
            return None

        df_window = df_window.copy()
        df_window['t_hours'] = (df_window[self.time_col] - global_reference_time).dt.total_seconds() / 3600.0

        return df_window

    def _perform_cosinor_analysis(self, df_window):
        if self.sensor_col not in df_window.columns:
            warnings.warn(f"{self.sensor_col} column not found in the data window.")
            return None

        y = df_window[self.sensor_col].values
        t = df_window['t_hours'].values
        omega = 2.0 * np.pi / self.period
        X = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])

        try:
            params, residuals, rank, s = lstsq(X, y, rcond=None)
        except Exception as e:
            warnings.warn(f"Error during cosinor analysis: {e}")
            return None

        M, A_coef, B_coef = params
        amplitude = np.sqrt(A_coef**2 + B_coef**2)
        phase_radian = math.atan2(B_coef, A_coef)
        phase_hours = (phase_radian / (2.0 * np.pi)) * self.period
        if phase_hours < 0:
            phase_hours += self.period
        return M, amplitude, phase_hours

    def run(self):
        results = []
        grouped = self.data.groupby(self.name_col)
        for name, group in tqdm(grouped, desc="Processing patients"):
            global_reference_time = group[self.time_col].min()
            start_analysis_time = global_reference_time + pd.Timedelta(hours=self.window_length_hours)
            valid_times = group[group[self.time_col] >= start_analysis_time][self.time_col]
            for ref_time in valid_times:
                df_window = self._prepare_window_data(group, ref_time, global_reference_time)
                if df_window is None:
                    M, amplitude, phase = (np.nan, np.nan, np.nan)
                else:
                    result = self._perform_cosinor_analysis(df_window)
                    if result is None:
                        M, amplitude, phase = (np.nan, np.nan, np.nan)
                    else:
                        M, amplitude, phase = result
                results.append({
                    self.name_col: name,
                    self.time_col: ref_time,
                    'MESOR': M,
                    'Amplitude': amplitude,
                    'Phase_hours': phase
                })
        return pd.DataFrame(results)

def PipelineSensor(df):
    df['targetTime'] = pd.to_datetime(df['targetTime'])
    processed_data = df.copy()
    # For demonstration, we run sensor analysis for ENMO and HR over two windows.
    cosinor_ENMO_48h = CosinorAnalyzer(data=df, window_length_hours=48, coverage_threshold=0.5, sensor='ENMO')
    cosinor_ENMO_week = CosinorAnalyzer(data=df, window_length_hours=24*7, coverage_threshold=0.5, sensor='ENMO')
    cosinor_ENMO_48h_data = cosinor_ENMO_48h.run()
    cosinor_ENMO_week_data = cosinor_ENMO_week.run()

    cosinor_HR_48h = CosinorAnalyzer(data=df, window_length_hours=48, coverage_threshold=0.5, sensor='HR')
    cosinor_HR_week = CosinorAnalyzer(data=df, window_length_hours=24*7, coverage_threshold=0.5, sensor='HR')
    cosinor_HR_48h_data = cosinor_HR_48h.run()
    cosinor_HR_week_data = cosinor_HR_week.run()

    cosinor_ENMO_merged = pd.merge(
        cosinor_ENMO_48h_data, cosinor_ENMO_week_data,
        on=['이름','targetTime'], suffixes=('_ENMO_48h', '_ENMO_week')
    )
    cosinor_HR_merged = pd.merge(
        cosinor_HR_48h_data, cosinor_HR_week_data,
        on=['이름','targetTime'], suffixes=('_HR_48h', '_HR_week')
    )
    cosinor_merged = pd.merge(
        cosinor_ENMO_merged, cosinor_HR_merged,
        on=['이름','targetTime']
    )
    final_df = pd.merge(
        processed_data, cosinor_merged,
        on=['이름','targetTime']
    )
    return final_df

# ----- Target Variable Creation -----
def create_binary_targets(df, target_col, periods, freq='15T'):
    for period, shift_steps in periods.items():
        df[f'{target_col}_{period}'] = (
            df.groupby('key')[target_col]
            .transform(lambda x: x.rolling(window=shift_steps, min_periods=1).max())
            .shift(-shift_steps)
            .fillna(0)
            .astype(int)
        )
    return df

# ----- Sequence Generation -----
def process_key(key, df, sequence_length, features, targets, static_features, max_shift):
    sequences = []
    patient_data = df[df['key'] == key].sort_values('targetTime').reset_index(drop=True)
    data_values = patient_data[features].values
    y = patient_data[targets].values

    if len(patient_data) < sequence_length + max_shift:
        return sequences, None

    static_x = patient_data[static_features].iloc[0].values
    hospital_name = extract_hospital_name(patient_data['idxInfo'].iloc[0])

    for i in range(len(patient_data) - sequence_length - max_shift):
        seq_x = data_values[i:i+sequence_length]
        seq_y = y[i+sequence_length]
        sequences.append((static_x, seq_x, seq_y))
    return sequences, hospital_name

def create_sequences_parallel(df, sequence_length, features, targets, static_features, max_shift):
    keys = df['key'].unique()
    sequences = []
    hospitals = []
    # Process keys sequentially by using one worker
    results = [process_key(key, df, sequence_length, features, targets, static_features, max_shift) for key in tqdm(keys, desc="Processing keys sequentially")]
    for seq_list, hospital_name in results:
        sequences.extend(seq_list)
        if hospital_name is not None:
            hospitals.extend([hospital_name] * len(seq_list))
    return sequences, hospitals


# =============================================================================
# 4. HOSPITAL-LEVEL PROCESSING (Modified to set num_augment=2 for 동국대)
# =============================================================================

# We will store the sequences for each hospital in a dictionary.
hospital_sequences = {}

# Loop through each hospital
for hosp in hospitals:
    print(f"\nProcessing Hospital: {hosp}")
    hosp_data = data[data['hospital'] == hosp].copy()

    # --- Print Class Distribution BEFORE Augmentation ---
    print(f"  Class Distribution for Hospital {hosp} BEFORE Augmentation:")
    print(hosp_data['dangerous_action'].value_counts(normalize=True)) # Print normalized distribution for better comparison

    # --- Conditional Augmentation ---
    num_augments_for_hosp = 3 # Default number of augmentations
    if hosp == "동국대":
        num_augments_for_hosp = 1 # Set num_augment to 2 for 동국대
        print(f"  Applying Data Augmentation for Hospital: {hosp} with num_augments=2")
    else:
        print(f"  Applying Data Augmentation for Hospital: {hosp} with default num_augments={num_augments_for_hosp}")

    augmentor = DataAugmentor(sigma=0.5, num_augments=num_augments_for_hosp) # Initialize DataAugmentor with hospital-specific num_augments
    hosp_data_aug = augmentor.augment_patient_data(hosp_data, continuous_columns, categorical_columns)

    # --- Print Class Distribution AFTER Augmentation ---
    print(f"  Class Distribution for Hospital {hosp} AFTER Augmentation:")
    print(hosp_data_aug['dangerous_action'].value_counts(normalize=True)) # Print normalized distribution for better comparison
    print("-" * 50) # Separator for clarity


    # --- Label Encoding --- (Apply label encoding regardless of augmentation)
    label_encoders = {}
    for col in categorical_columns:
        if col == "idxInfo":
            continue
        if hosp_data_aug[col].dtype == 'object' or hosp_data_aug[col].nunique() <= 20:
            le = LabelEncoder()
            hosp_data_aug[col] = le.fit_transform(hosp_data_aug[col].astype(str))
            label_encoders[col] = le

    # Sort by key and targetTime
    hosp_data_aug = hosp_data_aug.sort_values(by=['key', 'targetTime']).reset_index(drop=True)

    # --- Sensor Data Processing --- (Apply sensor processing regardless of augmentation)
    hosp_data_for_sensor = hosp_data_aug.rename(columns={'key': '이름'})
    hosp_data_proc = PipelineSensor(hosp_data_for_sensor)
    hosp_data_proc.rename(columns={'이름': 'key'}, inplace=True)

    # Drop extra sensor columns not needed further
    drop_columns_extra = [
        'Amplitude_HR_48h','Amplitude_ENMO_48h',
        'MESOR_HR_48h','MESOR_ENMO_48h',
        'Phase_hours_HR_48h','Phase_hours_ENMO_48h',
        'Normalized_Eight_Hour_Entropy','Eight_Hour_Entropy'
    ]
    hosp_data_proc.drop(columns=drop_columns_extra, inplace=True, errors='ignore')

    hosp_data_proc.rename(columns={'이름': 'key'}, inplace=True)

    # --- Target Variable Creation --- (Apply target creation regardless of augmentation)
    periods = {'1h': 4, '1d': 96, '1w': 672}
    hosp_data_proc = create_binary_targets(hosp_data_proc, 'dangerous_action', periods)
    hosp_data_proc = hosp_data_proc.fillna(0)

    # Define features and targets (keeping idxInfo)
    features = [col for col in hosp_data_proc.columns
                if col not in ['key', 'targetTime', 'dangerous_action',
                                                    'dangerous_action_1h', 'dangerous_action_1d', 'dangerous_action_1w']] + ['idxInfo']
    targets = ['dangerous_action_1h', 'dangerous_action_1d', 'dangerous_action_1w']
    static_features = [col for col in features if col in categorical_columns]
    sequential_features = [col for col in features if col not in static_features]

    # --- (Optional) Splitting Data --- (Apply train/test split regardless of augmentation)
    unique_keys = hosp_data_proc['key'].unique()
    train_keys, test_keys = train_test_split(unique_keys, test_size=0.2, random_state=42)
    hosp_train = hosp_data_proc[hosp_data_proc['key'].isin(train_keys)].reset_index(drop=True)
    hosp_test = hosp_data_proc[hosp_data_proc['key'].isin(test_keys)].reset_index(drop=True)

    # --- Sequence Generation --- (Apply sequence generation regardless of augmentation)
    sequence_length = 192  # e.g., past 192 time steps (15-min intervals)
    max_shift = 672      # maximum shift for 1-week prediction
    train_sequences, train_hospitals = create_sequences_parallel(
        df=hosp_train,
        sequence_length=sequence_length,
        features=sequential_features,
        targets=targets,
        static_features=static_features,
        max_shift=max_shift
    )
    test_sequences, test_hospitals = create_sequences_parallel(
        df=hosp_test,
        sequence_length=sequence_length,
        features=sequential_features,
        targets=targets,
        static_features=static_features,
        max_shift=max_shift
    )
    all_sequences = train_sequences + test_sequences
    # Since we are processing hospital by hospital, we already know the hospital name.
    hospital_sequences[hosp] = all_sequences
    print(f"Hospital {hosp}: Generated {len(all_sequences)} sequences.")


# =============================================================================
# 5. SAVE THE HOSPITAL-GROUPED SEQUENCES
# =============================================================================


# Create a directory to hold separate hospital files
output_dir = "processed_sequences_by_hospital"
os.makedirs(output_dir, exist_ok=True)

# Save each hospital's sequences to its own file
for hosp, sequences in hospital_sequences.items():
    file_name = f"{hosp}.pkl"  # e.g., "HospitalA.pkl"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(sequences, f)
    print(f"Saved {len(sequences)} sequences for hospital {hosp} to {file_path}")

print("\nHospital-level sequences saved separately.")