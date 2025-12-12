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
import matlab
import matlab.engine
import pandas as pd
import pickle
import sys

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
    # only keep these channels
    raw.pick(channels_to_include)
    # downsample to 100 Hz
    raw.resample(100, npad="auto")
    # bandpass filter between 0.4 to 30 Hz
    raw.filter(0.4, 30, fir_design="firwin")
    # apply common average reference montage
    raw = common_average_montage(raw, channels_to_include)

    # Sleep staging for C3, Cz, and C4
    sls_c3 = yasa.SleepStaging(raw, eeg_name="C3")
    predicted_c3 = sls_c3.predict()

    sls_cz = yasa.SleepStaging(raw, eeg_name="CZ")
    predicted_cz = sls_cz.predict()

    sls_c4 = yasa.SleepStaging(raw, eeg_name="C4")
    predicted_c4 = sls_c4.predict()
    
    # Determine the consensus stage
    consensus_stage = determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4)
    return consensus_stage

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

def get_percent_agreement(run_events, predicted_stages, window_start, stage_duration):
    # window_start, window_length, and stage_duration are in seconds
    # analogous to Dice coefficient, for agreement between two categorical discrete time series
    # Percent agreement = (number of stages in the prediction that match the manual stage at the corresponding time point / total number of epochs in the prediction) * 100%
    tolerance = stage_duration / 2  # stage predictions that occur within a tolerance of a change in manual stage are assigned the new manual stage for comparison
    # create list of time from start of window for each run event
    if run_events.shape[1] == 2:
        relative_event_times = [float(event) for event in run_events[:,0]]
    else:
        event_times = np.cumsum(run_events[:,1].astype(float))
        event_times = np.insert(event_times, 0, 0) # insert 0 as first element
        event_times = event_times[:-1] # remove last element
        # subtract window_start from each time so that a time of 0 corresponds to start of window
        relative_event_times = event_times - window_start
    #print([[time,stage] for time,stage in zip(relative_event_times, run_events[:,3])])
    # for each consensus stage, find the closest corresponding manual stage
    match_count = 0
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
                break
        #print(f"Predicted stage: {predicted_stages[i]}, Manual stage: {corresponding_manual_stage}")
    percent_agreement = (match_count / len(predicted_stages)) * 100
    return percent_agreement

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

else:
    # initialize dataframe to store results
    staging_methods_to_column_names = {
        'yasa': 'Percent_Agreement_YASA',
        'sleep_seeg': 'Percent_Agreement_SleepSEEG',
        'ad_ieeg': f'Percent_Agreement_AD_Ratio_iEEG_{threshold_ratios["ad_ieeg"]}',
        'ad_scalp': f'Percent_Agreement_AD_Ratio_scalp_{threshold_ratios["ad_scalp"]}',
    }

    column_names = [staging_methods_to_column_names[method] for method in auto_mode_staging_methods if method in staging_methods_to_column_names]
    results_df = pd.DataFrame(columns=['Subject', 'Run'] + column_names)

    # initialize rows with empty values for all patients and runs
    for param_dict in bids_path_list:
        subject = param_dict['subject']
        run = param_dict['run']
        result_row = {'Subject': subject, 'Run': run, 'Duration_seconds': np.nan}
        for method in param_dict['staging_methods']:
            if method in staging_methods_to_column_names:
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
                    method_column = staging_methods_to_column_names[method]
                    # exclude 'Average' row
                    valid_runs = patient_results[patient_results['Run'] != 'Average']
                    # calculate average weighted by run duration
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

    # open layout file and edit data format to fit mne requirements
    edit_made = False
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

    if edit_made:
        # write modified layout back to file
        with open(bids_path.fpath, 'wb') as f:
            f.writelines(lines)

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

    # read channels.tsv for this run to determine channel types
    channels_tsv_path = bids_path.copy().update(suffix="channels", extension=".tsv")
    channels_data = np.loadtxt(channels_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
    scalp_channel_names = channels_data[channels_data[:,1] == "EEG"][:,0].tolist()
    ieeg_channel_names = channels_data[channels_data[:,1] == "SEEG"][:,0].tolist()
    # convert electrode names to uppercase
    scalp_channel_names = [name.upper() for name in scalp_channel_names]
    ieeg_channel_names = [name.upper() for name in ieeg_channel_names]

    # read events.tsv in the same folder
    events_tsv_path = bids_path.copy().update(suffix="events", extension=".tsv", task=None, run=None)

    # get all events where event_category = "vigilance"
    events_data = np.loadtxt(events_tsv_path.fpath, dtype=str, delimiter="\t", skiprows=1)
    vigilance_events = events_data[events_data[:,2] == "vigilance"]

    # print test date of this run
    test_date = raw.info['meas_date'].strftime('%Y-%m-%d')
    test_time = raw.info['meas_date'].strftime('%H:%M:%S')
    test_duration = raw.n_times / raw.info['sfreq']
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

    # if run_events is empty or only has one entry, use comments from the lay file as run_events
    if len(run_events) <= 1:
        print(f"{len(run_events)} vigilance event(s) found in events.tsv for this run (<= 1). Using comments from layout file as vigilance events instead.")
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
                run_events = np.array([[line[0],line[4]] for line in comments_lines if line[4] in ['W','1','2','3','REM']])
                print(f"Found {len(run_events)} vigilance events from comments.")

    if stage_full_recording:
        window_start = 0
        window_stop = math.floor(test_duration)
        print("Staging full recording...")
    else:
        window_start = param_dict['window_start']
        window_stop = param_dict['window_stop']

    # update Duration_seconds in results dataframe
    results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run), 'Duration_seconds'] = test_duration

    # crop raw data to window_length seconds from window_start
    raw.crop(window_start, window_stop)

    # extract time from beginning and corresponding state from run_events
    if run_events.shape[1] == 2:
        # map vigilance state to corresponding code
        state_map = {
            'W': 'wake',
            '1': 'N1',
            '2': 'N2',
            '3': 'N3',
            'REM': 'REM'
        }
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
    y_dict = {'N3': 0, 'N2': 1, 'N1': 2, 'REM': 3, 'wake': 4, 'unknown': np.nan}
    event_y = [y_dict[state] for state in event_states]
    # convert to hours
    event_x = [time/3600 for time in event_times]

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

    print(f"Running automated sleep staging methods from {window_start/3600} to {(window_stop)/3600} hours... (entry {idx+1} of {len(bids_path_list)})")

    for method in staging_methods:
        print(f"Using staging method: {method}")
        # check if result already exists in the dataframe for this subject, run, and staging method
        if results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run), staging_methods_to_column_names[method]].values.size == 0:
            print(f"No entry found in results dataframe for subject {subject}, run {run}, method {method}. Proceeding to stage.")
            continue
        if not pd.isna(results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run), staging_methods_to_column_names[method]].values[0]):
            print(f"Result for subject {subject}, run {run}, method {method} already exists in results dataframe. Skipping this method.")
            continue
        if method == 'yasa':
            # set maximum run duration for YASA staging due to memory constraints
            run_duration_limit_for_yasa = 16 * 3600 # seconds
            if (window_stop - window_start) > run_duration_limit_for_yasa:
                print(f"Window length exceeds {run_duration_limit_for_yasa/3600} hours. Skipping due to memory constraints for YASA staging.")
                continue
            else:
                predicted_stages = get_yasa_consensus_stages(raw.copy())
        elif method[0:3] == 'ad_':
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
                predicted_stages, avg_ad_ratios, channel_ratios_by_epoch = get_alphadelta_stages(raw.copy(), picks, threshold_ratio)
                # save channel_ratios_by_epoch to pickle
                os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'ad_ratios'), exist_ok=True)
                with open(channel_ratios_path, 'wb') as f:
                    pickle.dump(channel_ratios_by_epoch, f)
                print(f"Saved channel alpha/delta ratios to {channel_ratios_path}.")

            # plot average alpha/delta ratios over time
            plt.figure(figsize=(10, 4))
            ad_x = np.arange(window_start, window_start + len(avg_ad_ratios)*30, 30) / 3600  # assuming 30-second epochs
            plt.plot(ad_x, avg_ad_ratios, marker=',')
            plt.xlabel(f'Time (hrs) (t=0 is {test_time})')
            plt.ylabel('Average Alpha/Delta Ratio')
            plt.title(f'Average alpha/delta ratios for {subject}, session {session}, task {task}, run {run} ({method})')
            plt.xlim(window_start/3600, (window_stop)/3600)
            plt.axhline(y=threshold_ratio, color='r', linestyle='--')
            plt.grid()
            
            if save_figures:
                # save to figures folder
                if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject, current_time)):
                    os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject, current_time), exist_ok=True)
                
                save_path = os.path.join(os.path.dirname(__file__), "figures", subject, current_time, f'{subject}_{session}_{task}_{run}_{window_start}_{window_stop}_avg_ad_ratios_{method}.png')
                plt.savefig(save_path)
                print(f"Saved average alpha/delta ratio figure to {save_path}.")
            
            if display_plots:
                plt.show()

        elif method == 'sleep_seeg':
            # check if EDF file already exists for this run
            edf_path = os.path.join(os.path.dirname(__file__), 'data', 'edf', f'{subject}_{session}_{task}_{run}_{window_start}_{window_stop}.edf')
            if os.path.exists(edf_path):
                print(f"Using existing EDF file at {edf_path} for SleepSEEG staging.")
            else:
                # write file as EDF format for SleepSEEG
                print("Exporting EDF file for SleepSEEG staging...")
                # create directory in data folder
                os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'edf'), exist_ok=True)
                mne.export.export_raw(edf_path, raw.copy().pick(picks=ieeg_channel_names).resample(200, npad="auto"), fmt='edf', overwrite=True)
                print(f"EDF file written to {edf_path} for SleepSEEG staging.")
            # start MATLAB engine
            eng = matlab.engine.start_matlab()
            # add SleepSEEG folder to MATLAB path
            eng.addpath(sleep_seeg_path)
            # call SleepSEEG function
            sleep_seeg_summary,sleep_seeg_stages = eng.SleepSEEG(edf_path, 0, nargout=2)
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

        print(f"Predicted {predicted_stages.count('W')} wake epochs and {len(predicted_stages) - predicted_stages.count('W')} sleep epochs using method {method}.")
        # calculate overlap coefficient between predicted_stages and manual stages
        percent_agreement = get_percent_agreement(run_events, predicted_stages, window_start, 30)
        print(f"Percent agreement between predicted stages and manual stages: {percent_agreement:.2f}%")

        # store result in dataframe
        result_row = {'Subject': subject, 'Run': run}
        if method in staging_methods_to_column_names:
            result_row[staging_methods_to_column_names[method]] = percent_agreement
            results_df.loc[(results_df['Subject'] == subject) & (results_df['Run'] == run), staging_methods_to_column_names[method]] = percent_agreement
            # overwrite results csv with new data
            results_df.to_csv(results_csv_path, index=False)
            print(f"Saved updated results to {results_csv_path}.")

        # x values for predicted stages
        stage_x = np.arange(window_start, window_start + len(predicted_stages)*30, 30) / 3600  # assuming 30-second epochs
        y_dict_prediction = {'N3': 0, 'N2': 1, 'N1': 2, 'R': 3, 'W': 4, 'sleep': np.nan, np.nan: np.nan}
        stage_y = [y_dict_prediction[stage] for stage in predicted_stages]

        print("Plotting hypnogram for cropped data...")
        plt.figure(figsize=(10, 4))
        plt.step(event_x, event_y, where='post', label='Manual Staging', color='tab:blue', alpha=0.5)
        plt.xlim(window_start/3600, (window_stop)/3600)
        plt.xlabel(f'Time (hrs) (t=0 is {test_time})')
        plt.ylabel('Vigilance State')
        plt.title(f'Cropped hypnogram for {subject}, session {session}, task {task}, run {run}')
        plt.ylim(-0.5, 4.5)
        plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'W'])
        plt.step(stage_x, stage_y, where='post', label=f'Staging: {method}, {percent_agreement:.2f}% agreement', color='tab:orange', alpha=0.5)
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

    # end timer
    end_time = time.time()
    total_processing_time += end_time - start_time
    print(f"Processing time = {timedelta(seconds=end_time-start_time)}")

print(f"Total processing time = {timedelta(seconds=total_processing_time)}")