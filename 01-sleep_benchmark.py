# 01-sleep_benchmark.py
# reads 01-config.yaml for parameters
# accepts an optional command-line argument for updating an existing results csv file
# USAGE: python 01-sleep_benchmark.py [path/to/existing_results.csv]

import os
import configparser
import matplotlib.pyplot as plt
import numpy as np
import time
import mne
import yaml
import math
from datetime import datetime
from datetime import timedelta
import yasa
from collections import Counter
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
    write_raw_bids
)
MATLAB_AVAILABLE = True
try:
    import matlab
    import matlab.engine
except ModuleNotFoundError as e:
    MATLAB_AVAILABLE = False
    print(f"ModuleNotFoundError: {e}")
    print("MATLAB engine for Python not available. SleepSEEG staging will be disabled.")
import pandas as pd
import pickle
import sys
import io

MERGE_EVENT_LISTS = True
#YASA_ELECTRODE = 'C3'  # electrode to use for YASA staging when not using consensus

def time_to_seconds(time_str):
    # converts a time string in the format HH:MM:SS to seconds
    h, m, s = map(float, time_str.split(":"))
    return h * 3600 + m * 60 + s

def get_earliest_date(tsv_path):
    # get earliest date from events.tsv file
    events_data = np.loadtxt(tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
    dates = [event[0].split("T")[0] for event in events_data]
    earliest_date = min(dates)
    return earliest_date

# Function to determine the consensus stage based on stages from C3, C4, and Cz
def determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4):
    consensus_stage = []
    for i in range(len(predicted_c3)):
        stages = [predicted_c3[i], predicted_cz[i], predicted_c4[i]]
        stage_counts = Counter(stages)
        if stage_counts.most_common(1)[0][1] >= 2:  # Check if the most common stage appears at least twice
            consensus_stage.append(stage_counts.most_common(1)[0][0])
        else:
            consensus_stage.append(np.nan)  # No consensus
    return consensus_stage

# Function to apply common average montage to selected channels
def common_average_montage(raw, channels_to_include):
    # Select the data from the specified channels
    raw.pick(channels_to_include)
    # Compute the average reference
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    return raw

def get_yasa_consensus_stages(raw):
    # Specify the channels to include in the analysis
    channels_to_include = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
    # get length of recording
    raw_duration = raw.times[-1]
    all_consensus_stages = []
    # break raw into epochs
    for k in range(0, int(raw_duration), 3600*8):
        start_sec = k
        end_sec = min(k + 3600*8, raw_duration)
        raw_epoch = raw.copy().crop(tmin=start_sec, tmax=end_sec)
        # load epoch data
        raw_epoch.load_data()
        # only keep these channels
        raw_epoch.pick(channels_to_include)
        # downsample to 100 Hz
        raw_epoch.resample(100, npad="auto")
        # bandpass filter between 0.4 to 30 Hz
        raw_epoch.filter(0.4, 30, fir_design="firwin", verbose=False)
        # apply common average reference montage
        raw_epoch = common_average_montage(raw_epoch, channels_to_include)

        # Sleep staging for C3, Cz, and C4
        sls_c3 = yasa.SleepStaging(raw_epoch, eeg_name="C3")
        predicted_c3 = sls_c3.predict()

        sls_cz = yasa.SleepStaging(raw_epoch, eeg_name="CZ")
        predicted_cz = sls_cz.predict()

        sls_c4 = yasa.SleepStaging(raw_epoch, eeg_name="C4")
        predicted_c4 = sls_c4.predict()
        
        # Determine the consensus stage
        consensus_stage = determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4)
        all_consensus_stages.extend(consensus_stage)

    return all_consensus_stages

def get_yasa_stages(raw, electrode_name):
    # Specify the channels to include in the analysis
    channels_to_include = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
    # get length of recording
    raw_duration = raw.times[-1]
    all_predicted_stages = []
    # break raw into epochs
    print(f"Using electrode {electrode_name}...")
    for k in range(0, int(raw_duration), 3600*8):
        start_sec = k
        end_sec = min(k + 3600*8, raw_duration)
        raw_epoch = raw.copy().crop(tmin=start_sec, tmax=end_sec)
        # only keep these channels
        raw_epoch.pick(channels_to_include)
        # downsample to 100 Hz
        raw_epoch.resample(100, npad="auto")
        # load epoch data
        raw_epoch.load_data()
        # bandpass filter between 0.4 to 30 Hz
        raw_epoch.filter(0.4, 30, fir_design="firwin", verbose=False)
        # apply common average reference montage
        raw_epoch = common_average_montage(raw_epoch, channels_to_include)

        # print number of channels, epoch length, and sampling rate
        print(f"Number of channels: {len(raw_epoch.ch_names)}, Epoch length (s): {raw_epoch.times[-1]-raw_epoch.times[0]}, Sampling rate (Hz): {raw_epoch.info['sfreq']}")
        print("Staging with YASA...")

        # YASA sleep staging
        sls = yasa.SleepStaging(raw_epoch, eeg_name=electrode_name)
        predicted = sls.predict()
        
        all_predicted_stages.extend(predicted)

    return all_predicted_stages

def get_yasa_stages_dict(raw, electrode_names):
    stages_dict = {}
    # preprocessing for YASA
    # Specify the channels to include in the analysis
    channels_to_include = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
    raw_epoch = raw.copy()
    raw_epoch.pick(channels_to_include)
    raw_epoch.resample(100, npad="auto")
    raw_epoch.load_data()
    raw_epoch.filter(0.4, 30, fir_design="firwin", verbose=False)
    raw_montage = common_average_montage(raw_epoch, channels_to_include)
    print(f"Number of channels: {len(raw_montage.ch_names)}, Recording length (s): {raw_montage.times[-1]-raw_montage.times[0]}, Sampling rate (Hz): {raw_montage.info['sfreq']}")
    print("Staging with YASA...")
    for electrode in electrode_names:
        print(f"Using electrode {electrode}...")
        sls = yasa.SleepStaging(raw_montage, eeg_name=electrode)
        predicted = sls.predict()
        stages_dict[electrode] = predicted
    return stages_dict

def yasa_dict_to_consensus_stages(yasa_staging_dict):
    predicted_c3 = yasa_staging_dict['C3']
    predicted_cz = yasa_staging_dict['CZ']
    predicted_c4 = yasa_staging_dict['C4']
    consensus_stages = determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4)
    return consensus_stages

def get_alphadelta_stages(raw, picks, threshold_ratio):
    # calculate alpha/delta ratio on each channel
    sfreq = raw.info['sfreq']
    epoch_length = 30  # seconds
    n_epochs = int(np.floor(raw.n_times / (sfreq * epoch_length)))
    channel_ratios_by_epoch = []
    avg_ratios = []
    for epoch in range(n_epochs):
        start_sample = int(epoch * epoch_length * sfreq)
        end_sample = int((epoch + 1) * epoch_length * sfreq)
        epoch_data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)
        # compute power spectral density
        psd, freqs = mne.time_frequency.psd_array_welch(epoch_data, sfreq=sfreq, fmin=0.5, fmax=12, n_fft=8192, verbose=False)
        # compute alpha power (8-12 Hz)
        alpha_power = np.trapezoid(psd[:,(freqs >= 8) & (freqs <= 12)], axis=1)
        # compute delta power (0.5-4 Hz)
        delta_power = np.trapezoid(psd[:,(freqs >= 0.5) & (freqs <= 4)], axis=1)
        # compute alpha/delta ratio
        ratios = np.divide(alpha_power, delta_power)
        # store ratios and channels for this epoch
        channel_to_ratio_dict = {channel: ratio for channel, ratio in zip(picks, ratios)}
        channel_ratios_by_epoch.append(channel_to_ratio_dict)
        # average across channels
        avg_ratio = np.nanmean(ratios)
        avg_ratios.append(avg_ratio)

    # average alpha/delta ratios across channels
    predicted_stages = []
    for ratio in avg_ratios:
        if ratio < threshold_ratio:
            predicted_stages.append('sleep')
        else:
            predicted_stages.append('W')
    return predicted_stages, avg_ratios, channel_ratios_by_epoch

def channel_ratios_to_stages(channel_ratios_by_epoch, threshold_ratio):
    predicted_stages = []
    avg_ratios = []
    for epoch_ratios in channel_ratios_by_epoch:
        avg_ratio = np.nanmean(list(epoch_ratios.values()))
        avg_ratios.append(avg_ratio)
        if avg_ratio < threshold_ratio:
            predicted_stages.append('sleep')
        else:
            predicted_stages.append('W')
    return predicted_stages, avg_ratios

def get_percent_agreement(run_events, predicted_stages, segment_start, stage_duration):
    # segment_start, window_length, and stage_duration are in seconds
    # analogous to Dice coefficient, for agreement between two categorical discrete time series
    # Percent agreement = (number of stages in the prediction that match the manual stage at the corresponding time point / total number of epochs in the prediction) * 100%
    # returns: overall percent agreement, wake/sleep percent agreement
    tolerance = stage_duration / 2  # stage predictions that occur within a tolerance of a change in manual stage are assigned the new manual stage for comparison
    # create list of time from start of window for each run event
    if run_events.shape[1] == 2:
        relative_event_times = [float(time)-segment_start for time in run_events[:,0]]
    else:
        event_times = np.cumsum(run_events[:,1].astype(float))
        event_times = np.insert(event_times, 0, 0) # insert 0 as first element
        event_times = event_times[:-1] # remove last element
        # subtract segment_start from each time so that a time of 0 corresponds to start of segment
        relative_event_times = event_times - segment_start
    #print([[time,stage] for time,stage in zip(relative_event_times, run_events[:,3])])
    # for each consensus stage, find the closest corresponding manual stage
    match_count = 0
    wake_sleep_match_count = 0
    wake_sleep_tp = 0
    wake_sleep_tn = 0
    wake_sleep_fp = 0
    wake_sleep_fn = 0
    nan_prediction_count = 0
    for i in range(len(predicted_stages)):
        corresponding_manual_stage = None
        predicted_stage_start = i * stage_duration
        for j in range(len(relative_event_times)-1):
            if (predicted_stage_start >= relative_event_times[j] - tolerance) and (predicted_stage_start < relative_event_times[j+1] - tolerance):
                #print(f"{predicted_stage_start} between {relative_event_times[j]} and {relative_event_times[j+1]}")
                if run_events.shape[1] == 2:
                    state_map = {
                        'W': 'wake',
                        '1': 'N1',
                        '2': 'N2',
                        '3': 'N3',
                        'REM': 'REM'
                    }
                    corresponding_manual_stage = state_map[run_events[j,1]]
                else:
                    corresponding_manual_stage = run_events[j,3]
                if ((isinstance(predicted_stages[i], str)) and ((predicted_stages[i].lower() == corresponding_manual_stage.lower()) or ((predicted_stages[i][0].lower() == corresponding_manual_stage[0].lower()) and (predicted_stages[i][0].lower() in ['r','w'])) or (predicted_stages[i] == 'sleep' and corresponding_manual_stage[0].lower() in ['n','r']))):
                    match_count += 1
                # check for wake/sleep match
                if ((isinstance(predicted_stages[i], str)) and (((predicted_stages[i][0].lower() == 'w') and (corresponding_manual_stage == 'wake')) or ((predicted_stages[i][0].lower() != 'w') and (corresponding_manual_stage in ['N1','N2','N3','REM'])))):
                    wake_sleep_match_count += 1
                    if (predicted_stages[i][0].lower() == 'w'):
                        wake_sleep_tp += 1
                    else:
                        wake_sleep_tn += 1
                else:
                    if (not (isinstance(predicted_stages[i], str))):
                        nan_prediction_count += 1
                    elif (predicted_stages[i][0].lower() == 'w'):
                        wake_sleep_fp += 1
                    else:
                        wake_sleep_fn += 1
                break
        #print(f"Predicted stage: {predicted_stages[i]}, Manual stage: {corresponding_manual_stage}")
    percent_agreement = (match_count / len(predicted_stages)) * 100
    wake_sleep_percent_agreement = (wake_sleep_match_count / len(predicted_stages)) * 100
    return percent_agreement, wake_sleep_percent_agreement, wake_sleep_tp, wake_sleep_tn, wake_sleep_fp, wake_sleep_fn, nan_prediction_count

# threshold ratios for different alpha/delta methods
threshold_ratios = {
    'ad_scalp': 0.02,
    'ad_ieeg': 0.05,
    'ad_all': 0.04
}

with open(os.path.join(os.path.dirname(__file__), "01-config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

bids_root = config['PARAMS']['bids_root']
sleep_seeg_path = config['PARAMS']['sleep_seeg_path']
bids_path_list = config['PARAMS']['bids_path_list']
display_plots = config['PARAMS']['display_plots']

auto_mode_enabled = config['PARAMS']['auto_mode_enabled']
auto_mode_staging_methods = config['PARAMS']['auto_mode_staging_methods']

if auto_mode_enabled:
    # construct bids_path_list automatically
    print("Auto mode enabled. Constructing bids_path_list based on BIDS root...")
    bids_path_list = []
    # find all subjects in bids_root
    subjects = get_entity_vals(bids_root, 'subject')
    for subject in subjects:
        sessions = get_entity_vals(os.path.join(bids_root, f'sub-{subject}'), 'session')
        for session in sessions:
            runs = get_entity_vals(os.path.join(bids_root, f'sub-{subject}', f'ses-{session}', 'ieeg'), 'run')
            for run in runs:
                param_dict = {
                    'subject': subject,
                    'session': session,
                    'datatype': 'ieeg',
                    'task': 'all',
                    'run': run,
                    'suffix': 'ieeg',
                    'extension': '.lay',
                    'staging_methods': auto_mode_staging_methods,
                    'stage_full_recording': True,
                    'save_figures': True
                }
                bids_path_list.append(param_dict)
    print(f"Found {len(subjects)} subjects and {len(bids_path_list)} runs in total.")
    print(f"Subjects: {subjects[0]}, {subjects[1]}... {subjects[-1]}")

total_processing_time = 0

# get current date and time for filename
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# check for results csv filename argument
if __name__ == "__main__":
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else None

# initialize dataframe to store results
staging_methods_to_column_names = {
    'yasa': {'wake_sleep': 'Wake_Sleep_Percent_Agreement_YASA', 'all_stages': 'Overall_Percent_Agreement_YASA'},
    'sleep_seeg': {'wake_sleep': 'Wake_Sleep_Percent_Agreement_SleepSEEG', 'all_stages': 'Overall_Percent_Agreement_SleepSEEG'},
    'ad_ieeg': f'Wake_Sleep_Percent_Agreement_AD_Ratio_iEEG_{threshold_ratios["ad_ieeg"]}',
    'ad_scalp': f'Wake_Sleep_Percent_Agreement_AD_Ratio_scalp_{threshold_ratios["ad_scalp"]}',
}

staging_methods_to_prediction_count_column_names = {
    'yasa': ['Wake_Sleep_YASA_TP', 'Wake_Sleep_YASA_TN', 'Wake_Sleep_YASA_FP', 'Wake_Sleep_YASA_FN', 'Wake_Sleep_YASA_NaN'],
    'sleep_seeg': ['Wake_Sleep_SleepSEEG_TP', 'Wake_Sleep_SleepSEEG_TN', 'Wake_Sleep_SleepSEEG_FP', 'Wake_Sleep_SleepSEEG_FN', 'Wake_Sleep_SleepSEEG_NaN'],
    'ad_ieeg': ['Wake_Sleep_AD_Ratio_iEEG_TP', 'Wake_Sleep_AD_Ratio_iEEG_TN', 'Wake_Sleep_AD_Ratio_iEEG_FP', 'Wake_Sleep_AD_Ratio_iEEG_FN', 'Wake_Sleep_AD_Ratio_iEEG_NaN'],
    'ad_scalp': ['Wake_Sleep_AD_Ratio_scalp_TP', 'Wake_Sleep_AD_Ratio_scalp_TN', 'Wake_Sleep_AD_Ratio_scalp_FP', 'Wake_Sleep_AD_Ratio_scalp_FN', 'Wake_Sleep_AD_Ratio_scalp_NaN']
}

# set segment duration to 8 hours
segment_duration = 3600*8

if csv_filename:
    # construct dataframe from existing csv
    print(f"Loading existing results from {csv_filename}...")
    try:
        results_df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error loading CSV file {csv_filename}: {e}.")
        sys.exit(1)
    print(f"Saving results to existing CSV file: {csv_filename}")
    results_csv_path = csv_filename

    # set threshold ratios based on existing column names
    for col in results_df.columns:
        if 'Percent_Agreement_AD_Ratio_iEEG_' in col:
            threshold_ratios['ad_ieeg'] = float(col.split('_')[-1])
            staging_methods_to_column_names['ad_ieeg'] = col
        elif 'Percent_Agreement_AD_Ratio_scalp_' in col:
            threshold_ratios['ad_scalp'] = float(col.split('_')[-1])
            staging_methods_to_column_names['ad_scalp'] = col

else:
    column_names = []
    for method in auto_mode_staging_methods:
        if method in staging_methods_to_column_names:
            column_names += list(staging_methods_to_column_names[method].values()) if isinstance(staging_methods_to_column_names[method], dict) else [staging_methods_to_column_names[method]]
        if method in staging_methods_to_prediction_count_column_names:
            column_names += staging_methods_to_prediction_count_column_names[method]

    # column_names = [staging_methods_to_column_names[method] for method in auto_mode_staging_methods if method in staging_methods_to_column_names]
    results_df = pd.DataFrame(columns=['Subject', 'Run', 'Segment'] + column_names + ['Duration_seconds'])

    # initialize rows with empty values for all patients and runs
    for idx, param_dict in enumerate(bids_path_list):
        subject = param_dict['subject']
        run = param_dict['run']
        # get run duration
        print(f"Reading data duration for subject {subject}, run {run} ({idx+1} of {len(bids_path_list)})...")
        # open layout file and edit data format to fit mne requirements
        edit_made = False
        bids_path = BIDSPath(root=bids_root, subject=subject, session='ieeg01', task='all', run=run, datatype='ieeg', suffix='ieeg', extension='.lay')
        try:
            with open(bids_path.fpath, 'rb') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line == b'BirthDate= \r\n' or line == b'BirthDate=\r\n':
                        # append dash
                        lines[i] = line.strip() + b'-' + b'\r\n'
                        edit_made = True
                        print(f"Edited BirthDate in layout file {bids_path.fpath} to avoid TypeError.")
                    elif b'TestDate=' in line and len(line.split(b"/")[-1]) == 2+2: # accounting for return characters
                        # convert to four digit year
                        testdate = line.split(b'=')[-1].strip().decode('utf-8')
                        testdate = datetime.strptime(testdate, "%m/%d/%y").strftime("%m/%d/%Y")
                        # replace line in file
                        lines[i] = b'TestDate=' + testdate.encode('utf-8') + b'\r\n'
                        edit_made = True
                        print(f"Edited TestDate in layout file {bids_path.fpath} to match four digit year format.")
                        break
        except FileNotFoundError as e:
            print(f"Error reading layout file: {e}")
            continue

        if edit_made:
            # write modified layout back to file
            with open(bids_path.fpath, 'wb') as f:
                f.writelines(lines)
        try:
            raw = mne.io.read_raw_persyst(bids_path, preload=False, verbose=False)
            raw_duration = raw.duration
        except FileNotFoundError as e:
            print(f"Error reading raw data: {e}")
            continue
        except RuntimeError as e:
            print(f"Error reading raw data: {e}")
            continue
        
        # divide run into 8-hour chunks
        for index, k in enumerate(range(0, int(raw_duration), segment_duration)):
            segment_start_sec = k
            segment_end_sec = min(k + segment_duration, raw_duration)
            result_row = {'Subject': subject, 'Run': run, 'Segment': int(index+1), 'Duration_seconds': segment_end_sec-segment_start_sec}
            for method in param_dict['staging_methods']:
                if method in staging_methods_to_column_names:
                    if isinstance(staging_methods_to_column_names[method], dict):
                        for col_name in staging_methods_to_column_names[method].values():
                            result_row[col_name] = np.nan
                    else:
                        result_row[staging_methods_to_column_names[method]] = np.nan
            results_df.loc[len(results_df)] = result_row

    # add empty rows for average percent agreement across runs for each patient at the end
    if auto_mode_enabled:
        unique_subjects = set([param_dict['subject'] for param_dict in bids_path_list])
        # sort unique_subjects
        unique_subjects = sorted(unique_subjects)
        for subject in unique_subjects:
            avg_row = {'Subject': subject, 'Run': 'Average'}
            for method in auto_mode_staging_methods:
                if method in staging_methods_to_column_names:
                    if isinstance(staging_methods_to_column_names[method], dict):
                        for col_name in staging_methods_to_column_names[method].values():
                            avg_row[col_name] = np.nan
                    else:
                        avg_row[staging_methods_to_column_names[method]] = np.nan
            results_df.loc[len(results_df)] = avg_row

    # output path for results csv
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    results_csv_path = os.path.join(os.path.dirname(__file__), "results", f'sleep_staging_benchmark_results_{current_time}.csv')
    print(f"Saving results to new CSV file: {results_csv_path}")

print(f"Initial dataframe:\n{results_df}")

last_patient = None

for idx, param_dict in enumerate(bids_path_list):
    subject = param_dict['subject']
    session = param_dict['session']
    datatype = param_dict['datatype']
    task = param_dict['task']
    run = param_dict['run']
    suffix = param_dict['suffix']
    extension = param_dict['extension']
    staging_methods = param_dict['staging_methods']
    stage_full_recording = param_dict['stage_full_recording']
    save_figures = param_dict['save_figures']

    if subject != last_patient:
        if ((last_patient is not None) and (auto_mode_enabled)):
            print(f"Calculating percent agreement across runs for patient {last_patient}...")
            patient_results = results_df[results_df['Subject'] == last_patient]
            for method in auto_mode_staging_methods:
                if method in staging_methods_to_column_names:
                    if isinstance(staging_methods_to_column_names[method], dict):
                        for stage_type, method_column in staging_methods_to_column_names[method].items():
                            # exclude 'Average' row
                            valid_runs = patient_results[patient_results['Run'] != 'Average']
                            # calculate the average percent agreement weighted by run duration, excluding NaN values
                            valid_runs = valid_runs.dropna(subset=[method_column, 'Duration_seconds'])
                            avg_percent_agreement = valid_runs[method_column].mul(valid_runs['Duration_seconds']).sum() / valid_runs['Duration_seconds'].sum()
                            results_df.loc[(results_df['Subject'] == last_patient) & (results_df['Run'] == 'Average'), method_column] = avg_percent_agreement
                            print(f"Average percent agreement for method {method} ({stage_type}) for patient {last_patient}: {avg_percent_agreement:.2f}%")
                    else:
                        method_column = staging_methods_to_column_names[method]
                        # exclude 'Average' row
                        valid_runs = patient_results[patient_results['Run'] != 'Average']
                        # calculate the average percent agreement weighted by run duration, excluding NaN values
                        valid_runs = valid_runs.dropna(subset=[method_column, 'Duration_seconds'])
                        avg_percent_agreement = valid_runs[method_column].mul(valid_runs['Duration_seconds']).sum() / valid_runs['Duration_seconds'].sum()
                        results_df.loc[(results_df['Subject'] == last_patient) & (results_df['Run'] == 'Average'), method_column] = avg_percent_agreement
                        print(f"Average percent agreement for method {method} for patient {last_patient}: {avg_percent_agreement:.2f}%")
            # overwrite results csv with new data
            results_df.to_csv(results_csv_path, index=False)
        print(f"Processing new patient: {subject}")

    last_patient = subject

    # BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        datatype=datatype,
        task=task,
        run=run,
        suffix=suffix,
        extension=extension,
        root=bids_root
    )

    print(repr(bids_path))

    # start timer
    start_time = time.time()

    try:
        raw = mne.io.read_raw_persyst(bids_path)
    except Exception as e:
        print(f"Error reading raw data from {bids_path.fpath}: {e}. Skipping this run.")
        continue

    for annot in raw.annotations:
        if annot["description"] in ['W', 'N1', 'N2', 'N3', 'REM']:  
            #print(f"{annot['onset']}, {annot['duration']}, {annot['description']}")
            continue

    # read events.tsv in the same folder
    events_tsv_path = bids_path.copy().update(suffix="events", extension=".tsv", task=None, run=None)

    # get all events where event_category = "vigilance"
    events_data = np.loadtxt(events_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
    vigilance_events = events_data[events_data[:,2] == "vigilance"]

    # print test date of this run
    test_date = raw.info['meas_date'].strftime('%Y-%m-%d')
    test_time = raw.info['meas_date'].strftime('%H:%M:%S')
    test_duration = raw.duration
    earliest_date = get_earliest_date(events_tsv_path)

    def date_to_seconds(date_str, earliest_date_str = earliest_date):
        # converts a date string in the format YYYY-MM-DD to seconds since the earliest date
        date = datetime.strptime(date_str, "%Y-%m-%d")
        earliest_date = datetime.strptime(earliest_date_str, "%Y-%m-%d")
        delta = date - earliest_date
        return delta.days * 86400

    # test year should be 2010 if earliest date year is 2010
    if "2013" in test_date and "2010" in earliest_date:
        test_date = test_date.replace("2013", "2010")
        print(f"Adjusted test date to match earliest date year.")
    print(f"Test date of this run: {test_date}. Test time: {test_time}. Test duration (s): {test_duration}")
    print(f"Earliest date in events.tsv: {earliest_date}")

    # get events for this run
    run_events = [event for event in vigilance_events if ((date_to_seconds(event[0].split("T")[0]) + time_to_seconds(event[0].split("T")[1]) > (date_to_seconds(test_date) + time_to_seconds(test_time))) and (date_to_seconds(event[0].split("T")[0]) + time_to_seconds(event[0].split("T")[1]) <= (date_to_seconds(test_date) + time_to_seconds(test_time) + test_duration)))]
    # merge to numpy array
    run_events = np.array(run_events)
    print(f"Found {len(run_events)} vigilance events from events.tsv for this run.")

    # obtain all entries under [Comments] in lay file
    with open(bids_path.fpath, 'rb') as f:
        lines = f.readlines()
        comments_start = None
        comments_end = None
        for i, line in enumerate(lines):
            if line.strip() == b'[Comments]':
                comments_start = i + 1
            elif comments_start is not None and line.startswith(b'['):
                comments_end = i
                break
        if comments_start is not None:
            if comments_end is None:
                comments_end = len(lines)
            comments_lines = lines[comments_start:comments_end]
            comments_lines = [line.decode('utf-8').strip().split(",") for line in comments_lines if line.strip() != b'']
            lay_run_events = np.array([[line[0],line[4]] for line in comments_lines if line[4] in ['W','1','2','3','REM']])
            print(f"Found {len(lay_run_events)} vigilance events from comments in layout file.")
        else:
            lay_run_events = np.array([])
            print("No [Comments] section found in layout file.")

    # assert that run_events and lay_run_events do not both contain two or more entries
    # assert (not (len(run_events) >= 2) and (len(lay_run_events) >= 2)), "Both events.tsv and layout file comments contain two or more vigilance events. Please investigate."

    # check if no vigilance events found in either source
    if (len(run_events) == 0) and (len(lay_run_events) == 0):
        print("No vigilance events found in either events.tsv or layout file comments. Skipping this run.")
        continue

    if MERGE_EVENT_LISTS:
        print("Merging vigilance events from events.tsv and layout file comments...")
        
        tsv_to_lay_state_map = {
            'wake': 'W',
            'N1': '1',
            'N2': '2',
            'N3': '3',
            'REM': 'REM'
        }

        run_event_times = []
        lay_run_event_times = []

        run_event_times = np.cumsum(run_events[:,1].astype(float))
        run_event_times = np.insert(run_event_times, 0, 0) # insert 0 as first element
        run_event_times = run_event_times[:-1] # remove last element
        run_event_states = [tsv_to_lay_state_map[state] for state in run_events[:,3]]

        lay_run_event_times = [float(event) for event in lay_run_events[:,0]]
        lay_run_event_states = lay_run_events[:,1]

        # zip times and states together and sort by time
        combined_events = list(zip(run_event_times, run_event_states)) + list(zip(lay_run_event_times, lay_run_event_states))
        combined_events.sort(key=lambda x: x[0])
        # remove duplicates (events with same time), keeping the first occurrence
        merged_event_times = []
        merged_event_states = []
        seen_times = []
        for t, state in combined_events:
            tolerance = 1 # one second tolerance for merging events
            if not any(abs(t - seen_time) <= tolerance for seen_time in seen_times):
                merged_event_times.append(t)
                merged_event_states.append(state)
                seen_times.append(t)
        event_times = merged_event_times
        event_states = merged_event_states

        print(f"Merged to {len(event_times)} vigilance events for this run.")
        # convert to numpy array
        run_events = np.array(list(zip(event_times, event_states)))
    
    else:
        # use only one of the two event lists
        if len(run_events) <= 2:
            print("Using vigilance events from layout file comments only.")
            run_events = lay_run_events
        else:
            print("Using vigilance events from events.tsv only.")

        state_map = {
            'W': 'wake',
            '1': 'N1',
            '2': 'N2',
            '3': 'N3',
            'REM': 'REM'
        }
        # extract time from beginning and corresponding state from run_events
        if run_events.shape[1] == 2:
            # map vigilance state to corresponding code
            event_times = [float(event) for event in run_events[:,0]]
            event_states = [state_map[state] for state in run_events[:,1]]
        else:
            # convert second column to cumulative sum 
            event_times = np.cumsum(run_events[:,1].astype(float))
            event_times = np.insert(event_times, 0, 0) # insert 0 as first element
            event_times = event_times[:-1] # remove last element
            event_states = run_events[:,3]
            # insert event at the end of the recording
            event_times = np.append(event_times, test_duration)
            event_states = np.append(event_states, event_states[-1])

    # map vigilance states to numerical values for plotting
    y_dict = {'N3': 0, 'N2': 1, 'N1': 2, 'REM': 3, 'wake': 4, 'W': 4, '1': 2, '2': 1, '3': 0, 'unknown': np.nan}
    event_y = [y_dict[state] for state in event_states]
    # convert to hours
    event_x = [t/3600 for t in event_times]

    # plot vigilance states over time
    print("Plotting hypnogram of entire run...")
    plt.figure(figsize=(10, 4))
    plt.step(event_x, event_y, where='post')
    plt.xlabel(f'Time (hrs) (t=0 is {test_time})')
    plt.ylabel('Vigilance State')
    plt.title(f'Hypnogram for {subject}, session {session}, task {task}, run {run}')
    plt.ylim(-0.5, 4.5)
    plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'W'])
    plt.grid()

    if save_figures:
        # save to figures folder
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject, current_time)):
            os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject, current_time), exist_ok=True)
        
        save_path = os.path.join(os.path.dirname(__file__), "figures", subject, current_time, f'{subject}_{session}_{task}_{run}_full_hypnogram.png')
        plt.savefig(save_path)
        print(f"Saved full hypnogram figure to {save_path}.")

    if display_plots:
        plt.show()

    # close figure
    plt.close()

    if stage_full_recording:
        window_start = 0
        window_stop = raw.times[-1]
        print("Staging full recording...")
    else:
        window_start = param_dict['window_start']
        window_stop = param_dict['window_stop']

    # crop raw data to window_length seconds from window_start
    # raw.crop(window_start, window_stop)

    print(f"Running automated sleep staging methods from {window_start/3600} to {(window_stop)/3600} hours... (entry {idx+1} of {len(bids_path_list)})")

    # # TODO: skip certain subjects and runs temporarily
    # if subject == 'umich0024':
    #     print(f"Skipping subject {subject}, run {run} due to memory issues.")
    #     continue

    for index, k in enumerate(range(window_start, int(window_stop), segment_duration)):
        segment_start_sec = k
        segment_end_sec = min(k + segment_duration, window_stop)
        print(f"\nStaging segment {index+1} from {segment_start_sec/3600} to {segment_end_sec/3600} hours...")
        
        for method in staging_methods:
            print(f"Using staging method: {method}")
            # check if result already exists in the dataframe for this subject, run, and staging method
            if isinstance(staging_methods_to_column_names[method], dict):
                this_entry_wake_sleep = results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), staging_methods_to_column_names[method]['wake_sleep']]
                this_entry_all_stages = results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), staging_methods_to_column_names[method]['all_stages']]
                if ((this_entry_wake_sleep.values.size != 0) and (not pd.isna(this_entry_wake_sleep.values[0]))) and ((this_entry_all_stages.values.size != 0) and (not pd.isna(this_entry_all_stages.values[0]))):
                    print(f"Results for subject {subject}, run {run}, segment {index+1}, method {method} already exist in results dataframe. Skipping this method.")
                    continue
            else:
                this_entry = results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), staging_methods_to_column_names[method]]
                if (this_entry.values.size != 0) and (not pd.isna(this_entry.values[0])):
                    print(f"Result for subject {subject}, run {run}, segment {index+1}, method {method} already exists in results dataframe. Skipping this method.")
                    continue

            this_method_timer_start = time.time()
            segment_raw = raw.copy().crop(tmin=segment_start_sec, tmax=segment_end_sec)

            if method == 'yasa':
                # check if pkl file with YASA staging results already exists for this segment
                yasa_results_path = os.path.join(os.path.dirname(__file__), 'data', 'yasa_staging', f'yasa_staging_{subject}_{session}_{task}_{run}_{segment_start_sec}_{segment_end_sec}.pkl')
                if os.path.exists(yasa_results_path):
                    print(f"Loading precomputed YASA staging results from {yasa_results_path}...")
                    with open(yasa_results_path, 'rb') as f:
                        yasa_staging_dict = pickle.load(f)
                else:
                    # if subject in ['umich0021','umich0022','umich0024']:
                    #     # TODO: temporarily skip subjects
                    #     print(f"Skipping YASA staging for subject {subject} due to memory issues.")
                    #     continue
                    yasa_staging_dict = get_yasa_stages_dict(segment_raw, ['C3','CZ','C4'])
                    # save to pickle
                    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'yasa_staging'), exist_ok=True)
                    with open(yasa_results_path, 'wb') as f:
                        pickle.dump(yasa_staging_dict, f)
                    print(f"Saved YASA staging results to {yasa_results_path}.")
                # print first 10 entries of each key in yasa_staging_dict
                print("YASA staging dictionary keys and first 10 entries:")
                for key in yasa_staging_dict:
                    print(f"{key}: {yasa_staging_dict[key][:10]}")
                print("Determining consensus stages...")
                predicted_stages = yasa_dict_to_consensus_stages(yasa_staging_dict)
            elif method[0:3] == 'ad_':
                # read channels.tsv for this run to determine channel types
                channels_tsv_path = bids_path.copy().update(suffix="channels", extension=".tsv")
                try:
                    channels_data = np.loadtxt(channels_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
                    ieeg_channel_names = channels_data[channels_data[:,1] == "SEEG"][:,0].tolist()
                    ieeg_channel_names = [name.upper() for name in ieeg_channel_names]
                    scalp_channel_names = channels_data[channels_data[:,1] == "EEG"][:,0].tolist()
                    scalp_channel_names = [name.upper() for name in scalp_channel_names]
                except Exception as e:
                    print(f"Error reading channels.tsv from {channels_tsv_path.fpath}: {e}")
                    print("Attempting to load channel names from other runs for this patient...")
                    # load channels_data from other runs for this patient
                    ieeg_channel_names_all_runs = []
                    scalp_channel_names_all_runs = []
                    # get all runs for this subject
                    subject_bids_paths = [p for p in bids_path_list if p['subject'] == subject]
                    for subject_bids_path in subject_bids_paths:
                        other_channels_tsv_path = BIDSPath(root=bids_root, subject=subject_bids_path['subject'], session='ieeg01', task='all', run=subject_bids_path['run'], datatype='ieeg', suffix='channels', extension='.tsv')
                        try:
                            other_channels_data = np.loadtxt(other_channels_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
                            other_ieeg_channel_names = other_channels_data[other_channels_data[:,1] == "SEEG"][:,0].tolist()
                            other_ieeg_channel_names = [name.upper() for name in other_ieeg_channel_names]
                            other_scalp_channel_names = other_channels_data[other_channels_data[:,1] == "EEG"][:,0].tolist()
                            other_scalp_channel_names = [name.upper() for name in other_scalp_channel_names]
                            ieeg_channel_names_all_runs.append(other_ieeg_channel_names)
                            scalp_channel_names_all_runs.append(other_scalp_channel_names)
                        except Exception as e2:
                            print(f"Error reading channels.tsv: {e2}\nSkipping to next .tsv file.")
                            continue
                    # if all entries of ieeg_channel_names_all_runs are identical, use the last entry
                    if (len(ieeg_channel_names_all_runs) > 0 and all(name_list == ieeg_channel_names_all_runs[0] for name_list in ieeg_channel_names_all_runs)) and (len(scalp_channel_names_all_runs) > 0 and all(name_list == scalp_channel_names_all_runs[0] for name_list in scalp_channel_names_all_runs)):
                        ieeg_channel_names = ieeg_channel_names_all_runs[0]
                        scalp_channel_names = scalp_channel_names_all_runs[0]
                        print(f"Using channel names from other runs for this patient as they are identical.")
                        print(f"iEEG channels: {ieeg_channel_names}")
                        print(f"Scalp EEG channels: {scalp_channel_names}")
                    else:
                        print(f"Unable to use channel names from other runs.")
                        print(f"ieeg_channel_names_all_runs: {ieeg_channel_names_all_runs}")
                        print(f"scalp_channel_names_all_runs: {scalp_channel_names_all_runs}")
                        print("Skipping alpha/delta staging for this segment.")
                        continue
                if method == 'ad_scalp':
                    picks = scalp_channel_names
                    threshold_ratio = threshold_ratios[method]
                elif method == 'ad_ieeg':
                    picks = ieeg_channel_names
                    threshold_ratio = threshold_ratios[method]
                elif method == 'ad_all':
                    picks = scalp_channel_names + ieeg_channel_names
                    threshold_ratio = threshold_ratios[method]
                else:
                    print(f"Method name {method} not recognized for alpha/delta staging. Defaulting to scalp EEG channels.")
                    picks = scalp_channel_names
                    threshold_ratio = threshold_ratios[method]
                # check if alpha/delta ratios have already been computed and saved for this run
                channel_ratios_path = os.path.join(os.path.dirname(__file__), 'data', 'ad_ratios', f'channel_ratios_{subject}_{session}_{task}_{run}_{window_start}_{window_stop}_{method}.pkl')
                if os.path.exists(channel_ratios_path):
                    print(f"Loading precomputed channel alpha/delta ratios from {channel_ratios_path}...")
                    with open(channel_ratios_path, 'rb') as f:
                        channel_ratios_by_epoch = pickle.load(f)
                    # get average alpha/delta ratios across channels
                    predicted_stages, avg_ad_ratios = channel_ratios_to_stages(channel_ratios_by_epoch, threshold_ratio)
                else:
                    # compute alpha/delta ratios and stages
                    # compute entire raw data for this run, as subsequent segments can load precomputed ratios
                    print(f"Computing alpha/delta ratios and stages for {method} for entire run (subsequent segments will load precomputed ratios)...")
                    predicted_stages, avg_ad_ratios, channel_ratios_by_epoch = get_alphadelta_stages(raw.copy(), picks, threshold_ratio)
                    # save channel_ratios_by_epoch to pickle
                    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'ad_ratios'), exist_ok=True)
                    with open(channel_ratios_path, 'wb') as f:
                        pickle.dump(channel_ratios_by_epoch, f)
                    print(f"Saved channel alpha/delta ratios to {channel_ratios_path}.")
                # use only this segment of predicted stages and average ad ratios
                predicted_stages = predicted_stages[segment_start_sec//30:(segment_start_sec+segment_duration)//30]
                avg_ad_ratios = avg_ad_ratios[segment_start_sec//30:(segment_start_sec+segment_duration)//30]

                # plot average alpha/delta ratios over time
                plt.figure(figsize=(10, 4))
                ad_x = np.arange(segment_start_sec, segment_start_sec + len(avg_ad_ratios)*30, 30) / 3600  # assuming 30-second epochs
                plt.plot(ad_x, avg_ad_ratios, marker=',')
                plt.xlabel(f'Time (hrs) (t=0 is {test_time})')
                plt.ylabel('Average Alpha/Delta Ratio')
                plt.title(f'Average alpha/delta ratios for {subject}, session {session}, task {task}, run {run}, segment {index+1} ({method})')
                plt.xlim(segment_start_sec/3600, (segment_end_sec)/3600)
                plt.axhline(y=threshold_ratio, color='r', linestyle='--')
                plt.grid()
                
                if save_figures:
                    # save to figures folder
                    if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject, current_time)):
                        os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject, current_time), exist_ok=True)
                    
                    save_path = os.path.join(os.path.dirname(__file__), "figures", subject, current_time, f'{subject}_{session}_{task}_{run}_{segment_start_sec}_{segment_end_sec}_avg_ad_ratios_{method}.png')
                    plt.savefig(save_path)
                    print(f"Saved average alpha/delta ratio figure to {save_path}.")
                
                if display_plots:
                    plt.show()
                
                # close figure
                plt.close()

            elif method == 'sleep_seeg':
                # TODO: skip certain subjects temporarily
                if subject in ['umich0024']:
                    print(f"Skipping SleepSEEG staging for subject {subject} due to EDF export issues.")
                    continue
                # check if EDF file already exists for this run
                edf_path = os.path.join(os.path.dirname(__file__), 'data', 'edf', f'{subject}_{session}_{task}_{run}_{segment_start_sec}_{segment_end_sec}.edf')
                if os.path.exists(edf_path):
                    print(f"Using existing EDF file at {edf_path} for SleepSEEG staging.")
                else:
                    # write file as EDF format for SleepSEEG
                    print("Exporting EDF file for SleepSEEG staging...")
                    # read channels.tsv for this run to determine channel types
                    channels_tsv_path = bids_path.copy().update(suffix="channels", extension=".tsv")
                    try:
                        channels_data = np.loadtxt(channels_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
                        ieeg_channel_names = channels_data[channels_data[:,1] == "SEEG"][:,0].tolist()
                        ieeg_channel_names = [name.upper() for name in ieeg_channel_names]
                    except Exception as e:
                        print(f"Error reading channels.tsv from {channels_tsv_path.fpath}: {e}")
                        print("Attempting to load iEEG channel names from other runs for this patient...")
                        # load channels_data from other runs for this patient
                        ieeg_channel_names_all_runs = []
                        # get all runs for this subject
                        subject_bids_paths = [p for p in bids_path_list if p['subject'] == subject]
                        for subject_bids_path in subject_bids_paths:
                            other_channels_tsv_path = BIDSPath(root=bids_root, subject=subject_bids_path['subject'], session='ieeg01', task='all', run=subject_bids_path['run'], datatype='ieeg', suffix='channels', extension='.tsv')
                            try:
                                other_channels_data = np.loadtxt(other_channels_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
                                other_ieeg_channel_names = other_channels_data[other_channels_data[:,1] == "SEEG"][:,0].tolist()
                                other_ieeg_channel_names = [name.upper() for name in other_ieeg_channel_names]
                                ieeg_channel_names_all_runs.append(other_ieeg_channel_names)
                            except Exception as e2:
                                print(f"Error reading channels.tsv: {e2}\nSkipping to next .tsv file.")
                                continue
                        # if all entries of ieeg_channel_names_all_runs are the identical, use the last entry
                        if len(ieeg_channel_names_all_runs) > 0 and all(name_list == ieeg_channel_names_all_runs[0] for name_list in ieeg_channel_names_all_runs):
                            ieeg_channel_names = ieeg_channel_names_all_runs[0]
                            print(f"Using iEEG channel names from other runs for this patient as they are identical: {ieeg_channel_names}")
                        else:
                            print(f"Unable to use iEEG channel names from other runs: {ieeg_channel_names_all_runs}\nSkipping SleepSEEG staging for this segment.")
                            continue
                    # create directory in data folder
                    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'edf'), exist_ok=True)
                    segment_raw_for_edf = segment_raw.copy().pick(picks=ieeg_channel_names).resample(200, npad="auto")
                    mne.export.export_raw(edf_path, segment_raw_for_edf, fmt='edf', overwrite=True)
                    print(f"EDF file written to {edf_path} for SleepSEEG staging.")
                if not MATLAB_AVAILABLE:
                    print("MATLAB engine for Python not available. Skipping SleepSEEG staging.")
                    continue
                # start MATLAB engine
                eng = matlab.engine.start_matlab()
                # add SleepSEEG folder to MATLAB path
                eng.addpath(sleep_seeg_path)
                # call SleepSEEG function
                out = io.StringIO()
                sleep_seeg_summary,sleep_seeg_stages = eng.SleepSEEG(edf_path, 0, nargout=2, stdout=out)
                # convert MATLAB cell array to Python list
                sleep_seeg_summary = [str(stage) for stage in sleep_seeg_summary]
                #print(f"SleepSEEG predicted stages: {predicted_stages}")
                # reshape 1D list to 2D matrix
                # get position of 'Date' string in stages_matlab
                extracted_stages = sleep_seeg_summary[sleep_seeg_summary.index('Sleep stage')+1:sleep_seeg_summary.index('# of epochs')]
                extracted_epoch_counts = sleep_seeg_summary[sleep_seeg_summary.index('# of epochs')+1:]
                #print(f"Extracted stages: {extracted_stages}")
                #print(f"Extracted epoch counts: {extracted_epoch_counts}")
                # construct predicted_stages by repeating each stage by its epoch count
                predicted_stages = []
                for stage, count in zip(extracted_stages, extracted_epoch_counts):
                    predicted_stages.extend([stage] * int(float(count)))
                eng.quit()
                # remove temporary EDF file
                # os.remove(edf_path)

            else:
                print(f"Staging method {method} not recognized. Skipping...")
                continue

            # stop timer
            this_method_timer_end = time.time()
            method_time_elapsed = this_method_timer_end - this_method_timer_start
            print(f"Predicted {predicted_stages.count('W')} wake epochs and {len(predicted_stages) - predicted_stages.count('W') - predicted_stages.count(np.nan)} sleep epochs using method {method} ({predicted_stages.count(np.nan)} epochs unable to be staged.) Processing time for this segment and method: {timedelta(seconds=method_time_elapsed)}")
            # calculate overlap coefficient between predicted_stages and manual stages
            overall_percent_agreement, wake_sleep_percent_agreement, tp, tn, fp, fn, nan_prediction_count = get_percent_agreement(run_events, predicted_stages, segment_start_sec, 30)
            print(f"Overall percent agreement between predicted stages and manual stages: {overall_percent_agreement:.2f}%")
            print(f"Wake/sleep percent agreement between predicted stages and manual stages: {wake_sleep_percent_agreement:.2f}%")
            print(f"Confusion matrix: (positive class is wake):")
            tp_string = f"TP = {tp}"
            fn_string = f"FN = {fn}"
            bar_position = max(len(tp_string), len(fn_string)) + 1
            matrix_top_row = f"{tp_string}{' ' * (bar_position - len(tp_string))}| FP = {fp}"
            print(matrix_top_row)
            print("-" * len(matrix_top_row))
            matrix_bottom_row = f"{fn_string}{' ' * (bar_position - len(fn_string))}| TN = {tn}"
            print(matrix_bottom_row)
            print(f"Number of epochs with NaN prediction: {nan_prediction_count}")
            print(f"Sensitivity: {tp / (tp + fn) if (tp + fn) > 0 else 0:.2f}")
            print(f"Specificity: {tn / (tn + fp) if (tn + fp) > 0 else 0:.2f}")
            print(f"PPV: {tp / (tp + fp) if (tp + fp) > 0 else 0:.2f}")
            print(f"NPV: {tn / (tn + fn) if (tn + fn) > 0 else 0:.2f}")
            print(f"F1 score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.2f}")

            # store result in dataframe
            #result_row = {'Subject': subject, 'Run': run}
            if method in staging_methods_to_column_names:
                if isinstance(staging_methods_to_column_names[method], dict):
                    results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), staging_methods_to_column_names[method]['all_stages']] = overall_percent_agreement
                    results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), staging_methods_to_column_names[method]['wake_sleep']] = wake_sleep_percent_agreement
                else:
                    results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), staging_methods_to_column_names[method]] = overall_percent_agreement
                # for each column in staging_methods_to_prediction_count_column_names
                for col in staging_methods_to_prediction_count_column_names[method]:
                    if col[-2:] == 'TP':
                        results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), col] = tp
                    elif col[-2:] == 'TN':
                        results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), col] = tn
                    elif col[-2:] == 'FP':
                        results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), col] = fp
                    elif col[-2:] == 'FN':
                        results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), col] = fn
                    elif col[-3:] == 'NaN':
                        results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run) & (results_df['Segment'] == index+1), col] = nan_prediction_count
                # overwrite results csv with new data
                results_df.to_csv(results_csv_path, index=False)
                print(f"Saved updated results to {results_csv_path}.")

            # x values for predicted stages
            stage_x = np.arange(segment_start_sec, segment_start_sec + len(predicted_stages)*30, 30) / 3600  # assuming 30-second epochs
            y_dict_prediction = {'N3': 0, 'N2': 1, 'N1': 2, 'R': 3, 'W': 4, 'sleep': np.nan, np.nan: np.nan}
            stage_y = [y_dict_prediction[stage] for stage in predicted_stages]

            print("Plotting hypnogram for cropped data...")
            plt.figure(figsize=(10, 4))
            plt.step(event_x, event_y, where='post', label='Manual Staging', color='tab:blue', alpha=0.5)
            plt.xlim(segment_start_sec/3600, (segment_end_sec)/3600)
            plt.xlabel(f'Time (hrs) (t=0 is {test_time})')
            plt.ylabel('Vigilance State')
            plt.title(f'Cropped hypnogram for {subject}, session {session}, task {task}, run {run}')
            plt.ylim(-0.5, 4.5)
            plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'W'])
            plt.step(stage_x, stage_y, where='post', label=f'Staging: {method}, {overall_percent_agreement:.2f}% agreement', color='tab:orange', alpha=0.5)
            plt.legend()
            plt.grid()
            
            if save_figures:
                # create a subdirectory in figures/patient/ with the current time
                if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject, current_time)):
                    os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject, current_time), exist_ok=True)
                
                save_path = os.path.join(os.path.dirname(__file__), "figures", subject, current_time, f'{subject}_{session}_{task}_{run}_{window_start}_{window_stop}_hypnogram_{method}.png')
                plt.savefig(save_path)
                print(f"Saved cropped hypnogram figure to {save_path}.")
            
            if display_plots:
                plt.show()
            
            # close figure
            plt.close()

    # end timer
    end_time = time.time()
    total_processing_time += end_time - start_time
    print(f"Patient processing time = {timedelta(seconds=end_time-start_time)}")

print(f"Total processing time = {timedelta(seconds=total_processing_time)}")