# 01-sleep_benchmark.py

import os
import configparser
import matplotlib.pyplot as plt
import numpy as np
import time
import mne
import yaml
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
)

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
    sls_c3 = yasa.SleepStaging(raw, eeg_name="C3") #['W' 'W' 'W' 'R' 'R' 'R' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W']
    predicted_c3 = sls_c3.predict()

    sls_cz = yasa.SleepStaging(raw, eeg_name="CZ") #['N2' 'N2' 'N2' 'N1' 'N2' 'N1' 'W' 'W' 'W' 'W' 'W' 'N1' 'R' 'R' 'R' 'R']
    predicted_cz = sls_cz.predict()

    sls_c4 = yasa.SleepStaging(raw, eeg_name="C4") #['W' 'W' 'W' 'W' 'N2' 'R' 'R' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W' 'W']
    predicted_c4 = sls_c4.predict()
    
    # Determine the consensus stage
    consensus_stage = determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4)
    return consensus_stage

def get_alphadelta_stages(raw, picks, threshold_ratio):
    # calculate alpha/delta ratio on each channel
    sfreq = raw.info['sfreq']
    epoch_length = 30  # seconds
    n_epochs = int(np.floor(raw.n_times / (sfreq * epoch_length)))
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
    return predicted_stages, avg_ratios

def get_percent_agreement(run_events, predicted_stages, window_start, stage_duration):
    # window_start, window_length, and stage_duration are in seconds
    # analogous to Dice coefficient, for agreement between two categorical discrete time series
    # Percent agreement = (number of stages in the prediction that match the manual stage at the corresponding time point / total number of epochs in the prediction) * 100%
    tolerance = stage_duration / 2  # stage predictions that occur within a tolerance of a change in manual stage are assigned the new manual stage for comparison
    # create list of time from start of window for each run event
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
                corresponding_manual_stage = run_events[j,3]
                if ((isinstance(predicted_stages[i], str)) and ((predicted_stages[i].lower() == corresponding_manual_stage.lower()) or ((predicted_stages[i][0].lower() == corresponding_manual_stage[0].lower()) and (predicted_stages[i][0].lower() in ['r','w'])) or (predicted_stages[i] == 'sleep' and corresponding_manual_stage[0].lower() in ['n','r']))):
                    match_count += 1
                break
        #print(f"Predicted stage: {predicted_stages[i]}, Manual stage: {corresponding_manual_stage}")
    percent_agreement = (match_count / len(predicted_stages)) * 100
    return percent_agreement

# config = configparser.ConfigParser()
# config.read(os.path.join(os.path.dirname(__file__), "01-config.ini"))
# print(config.sections())
# bids_root = config['PARAMS']['bids_root']
# bids_path_list = config['PARAMS']['bids_path_list']

with open(os.path.join(os.path.dirname(__file__), "01-config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

bids_root = config['PARAMS']['bids_root']
bids_path_list = config['PARAMS']['bids_path_list']

total_processing_time = 0

for param_dict in bids_path_list:
    # start timer
    start_time = time.time()

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

    raw = mne.io.read_raw_persyst(bids_path)

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

    # if subject = "umich0020", test year should be 2010
    if subject == "umich0020":
        test_date = test_date.replace("2013", "2010")
    print(f"Test date of this run: {test_date}. Test time: {test_time}. Test duration (s): {test_duration}")
    print(f"Earliest date in events.tsv: {earliest_date}")

    # get events for this run
    run_events = [event for event in vigilance_events if ((date_to_seconds(event[0].split("T")[0]) + time_to_seconds(event[0].split("T")[1]) > (date_to_seconds(test_date) + time_to_seconds(test_time))) and (date_to_seconds(event[0].split("T")[0]) + time_to_seconds(event[0].split("T")[1]) <= (date_to_seconds(test_date) + time_to_seconds(test_time) + test_duration)))]
    # merge to numpy array
    run_events = np.array(run_events)

    if stage_full_recording:
        window_start = 0
        window_stop = test_duration
        print("Staging full recording...")
    else:
        window_start = param_dict['window_start']
        window_stop = param_dict['window_stop']

    # crop raw data to window_length seconds from window_start
    raw.crop(window_start, window_stop)

    # extract time from beginning and corresponding state from run_events
    # convert second column to cumulative sum
    event_times = np.cumsum(run_events[:,1].astype(float))
    event_times = np.insert(event_times, 0, 0) # insert 0 as first element
    event_times = event_times[:-1] # remove last element
    event_states = run_events[:,3]

    # map vigilance states to numerical values for plotting
    y_dict = {'N3': 0, 'N2': 1, 'N1': 2, 'REM': 3, 'wake': 4}
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
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject)):
            os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject), exist_ok=True)
        
        plt.savefig(os.path.join(os.path.dirname(__file__), "figures", subject, f'{subject}_{session}_{task}_{run}_full_hypnogram.png'))

    plt.show()

    print(f"Running automated sleep staging methods from {window_start/3600} to {(window_stop)/3600} hours...")

    for method in staging_methods:
        print(f"Using staging method: {method}")
        if method == 'yasa':
            predicted_stages = get_yasa_consensus_stages(raw.copy())
        elif method[0:3] == 'ad_':
            if method == 'ad_scalp':
                picks = scalp_channel_names
            elif method == 'ad_ieeg':
                picks = ieeg_channel_names
            elif method == 'ad_all':
                picks = scalp_channel_names + ieeg_channel_names
            else:
                print(f"Method name {method} not recognized for alpha/delta staging. Defaulting to scalp EEG channels.")
                picks = scalp_channel_names
            threshold_ratio = 0.25  # threshold ratio to separate sleep vs wake    
            predicted_stages, avg_ad_ratios = get_alphadelta_stages(raw.copy(), picks, threshold_ratio)
            # plot average alpha/delta ratios over time
            plt.figure(figsize=(10, 4))
            ad_x = np.arange(window_start, window_start + len(avg_ad_ratios)*30, 30) / 3600  # assuming 30-second epochs
            plt.plot(ad_x, avg_ad_ratios, marker='o')
            plt.xlabel(f'Time (hrs) (t=0 is {test_time})')
            plt.ylabel('Average Alpha/Delta Ratio')
            plt.title(f'Average alpha/delta ratios for {subject}, session {session}, task {task}, run {run} ({method})')
            plt.xlim(window_start/3600, (window_stop)/3600)
            plt.axhline(y=threshold_ratio, color='r', linestyle='--')
            plt.grid()
            
            if save_figures:
                # save to figures folder
                if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject)):
                    os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject), exist_ok=True)
                
                plt.savefig(os.path.join(os.path.dirname(__file__), "figures", subject, f'{subject}_{session}_{task}_{run}_{window_start}_{window_stop}_avg_ad_ratios_{method}.png'))
            plt.show()
        else:
            print(f"Staging method {method} not recognized. Skipping...")
            continue

        print(f"Predicted stages: {predicted_stages}")
        # calculate overlap coefficient between predicted_stages and manual stages
        percent_agreement = get_percent_agreement(run_events, predicted_stages, window_start, 30)
        print(f"Percent agreement between predicted stages and manual stages: {percent_agreement:.2f}%")

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
            # save to figures folder
            if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", subject)):
                os.makedirs(os.path.join(os.path.dirname(__file__), "figures", subject), exist_ok=True)
            
            plt.savefig(os.path.join(os.path.dirname(__file__), "figures", subject, f'{subject}_{session}_{task}_{run}_{window_start}_{window_stop}_hypnogram_{method}.png'))
        plt.show()
    # end timer
    end_time = time.time()
    total_processing_time += end_time - start_time
    print(f"Processing time = {timedelta(seconds=end_time-start_time)}")

print(f"Total processing time = {timedelta(seconds=total_processing_time)}")