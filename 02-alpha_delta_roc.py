# 02-alpha_delta_roc.py
# Calculate alpha/delta ratios from iEEG and scalp EEG channels and plot ROC curves/histograms
# Uses parameters from 02-config.yaml or a specified .npz file containing precomputed ratios and true stages
# Usage: python 02-alpha_delta_roc.py [optional argument: path to .npz file]

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
import mne
import yaml
import math
from datetime import datetime
from datetime import timedelta
from collections import Counter
import sys
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
    write_raw_bids
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

def get_alphadelta_ratios(raw, picks):
    # calculate alpha/delta ratio on each channel
    sfreq = raw.info['sfreq']
    epoch_length = 30  # seconds
    n_epochs = int(np.floor(raw.n_times / (sfreq * epoch_length)))
    avg_ratios = []
    print("Number of epochs:", n_epochs)
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
        if (epoch + 1) % 500 == 0:
            print(f"Processed {epoch + 1}/{n_epochs} epochs")

    return avg_ratios

def get_corresponding_manual_stages(run_events, window_start, window_length, stage_duration):
    # window_start, window_length, and stage_duration are in seconds
    tolerance = stage_duration / 2  # stage predictions that occur within a tolerance of a change in manual stage are assigned the new manual stage for comparison
    # create list of time from start of window for each run event
    event_times = np.cumsum(run_events[:,1].astype(float))
    event_times = np.insert(event_times, 0, 0) # insert 0 as first element
    event_times = event_times[:-1] # remove last element
    # subtract window_start from each time so that a time of 0 corresponds to start of window
    relative_event_times = event_times - window_start
    #print([[time,stage] for time,stage in zip(relative_event_times, run_events[:,3])])
    # for each consensus stage, find the closest corresponding manual stage
    manual_stages = []
    for i in range(int(window_length / stage_duration)):
        corresponding_manual_stage = None
        predicted_stage_start = i * stage_duration
        for j in range(len(relative_event_times)-1):
            if (predicted_stage_start >= relative_event_times[j] - tolerance) and (predicted_stage_start < relative_event_times[j+1] - tolerance):
                corresponding_manual_stage = run_events[j,3]
        manual_stages.append(corresponding_manual_stage)
    return manual_stages

# config = configparser.ConfigParser()
# config.read(os.path.join(os.path.dirname(__file__), "01-config.ini"))
# print(config.sections())
# bids_root = config['PARAMS']['bids_root']
# bids_path_list = config['PARAMS']['bids_path_list']

with open(os.path.join(os.path.dirname(__file__), "02-config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

bids_root = config['PARAMS']['bids_root']
bids_path_list = config['PARAMS']['bids_path_list']

total_processing_time = 0

ad_ratio_scalp = []
ad_ratio_ieeg = []
true_stage = []

if __name__ == "__main__":
    npz_filename = sys.argv[1] if len(sys.argv) > 1 else None

if npz_filename is None:
    print("Using 02-config.yaml parameters for ROC calculation...")
    for idx, param_dict in enumerate(bids_path_list):
        # start timer
        start_time = time.time()

        subject = param_dict['subject']
        session = param_dict['session']
        datatype = param_dict['datatype']
        task = param_dict['task']
        run = param_dict['run']
        suffix = param_dict['suffix']
        extension = param_dict['extension']
        stage_full_recording = param_dict['stage_full_recording']

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
                if line == b'BirthDate= \r\n':
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

        raw = mne.io.read_raw_persyst(bids_path)

        for annot in raw.annotations:
            if annot["description"] in ['W', 'N1', 'N2', 'N3', 'REM']:  
                #print(f"{annot['onset']}, {annot['duration']}, {annot['description']}")
                continue

        # read channels.tsv for this run to determine channel types
        channels_tsv_path = bids_path.copy().update(suffix="channels", extension=".tsv")
        # check if file exists before loading
        if not os.path.exists(channels_tsv_path.fpath):
            print(f"Channels file {channels_tsv_path.fpath} not found. Skipping this run...")
            continue
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

        # if run_events is empty, skip this run
        if run_events.shape[0] == 0:
            print(f"No vigilance events found for this run. Skipping...")
            continue

        if stage_full_recording:
            window_start = 0
            window_stop = math.floor(test_duration)
            print("Processing full recording...")
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

        print(f"Reading alpha/delta ratios and corresponding sleep/wake label from {window_start/3600} to {(window_stop)/3600} hours... (entry {idx+1} of {len(bids_path_list)})")

        # get corresponding manual stages for this window
        manual_stages = get_corresponding_manual_stages(run_events, window_start, window_stop - window_start, 30)
        true_stage.extend(manual_stages)
        print(f"Number of manual stages for this window: {len(manual_stages)}")
        n_epochs = int(np.floor(raw.n_times / (raw.info['sfreq'] * 30)))
        print(f"Number of 30-second epochs in this window: {n_epochs}")
        # assert that lengths match
        assert len(manual_stages) == n_epochs, f"Length mismatch between manual stages ({len(manual_stages)}) and number of epochs ({n_epochs})"

        for picks, label in zip([scalp_channel_names, ieeg_channel_names] , ["Scalp EEG", "iEEG"]):
            print(f"Calculating average alpha/delta ratios for {label} channels...")
            avg_ad_ratios = get_alphadelta_ratios(raw.copy(), picks)

            if picks == scalp_channel_names:
                ad_ratio_scalp.extend(avg_ad_ratios)
            elif picks == ieeg_channel_names:
                ad_ratio_ieeg.extend(avg_ad_ratios)

        # end timer
        end_time = time.time()
        total_processing_time += end_time - start_time
        print(f"Processing time = {timedelta(seconds=end_time-start_time)}")

    # get current date and time for filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # create data directories if they don't exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "ad_ratios")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "data", "ad_ratios"), exist_ok=True)

    # save ad_ratio_scalp, ad_ratio_ieeg, and true_stage to combined npy file
    np.savez(os.path.join(os.path.dirname(__file__), "data", "ad_ratios", f"{bids_path_list[0]['subject']}_run{bids_path_list[0]['run']}_to_{bids_path_list[-1]['subject']}_run{bids_path_list[-1]['run']}_ad_ratios_{current_time}.npz"),
            ad_ratio_scalp=ad_ratio_scalp,
            ad_ratio_ieeg=ad_ratio_ieeg,
            true_stage=true_stage)
else:
    # attempt to open npz
    try:
        data = np.load(npz_filename, allow_pickle=True)
        ad_ratio_scalp = data['ad_ratio_scalp'].tolist()
        ad_ratio_ieeg = data['ad_ratio_ieeg'].tolist()
        true_stage = data['true_stage'].tolist()
        print(f"Loaded alpha/delta ratios and true stages from {npz_filename}.")
    except Exception as e:
        print(f"Error loading {npz_filename}: {e}")
        sys.exit(1)
    print("Using .npz file for ROC plotting instead of 02-config.yaml parameters.")
    # get current date and time for filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# plot ROC curves for scalp and ieeg
for ad_ratios, label in zip([ad_ratio_scalp, ad_ratio_ieeg], ['Scalp EEG', 'iEEG']):
    # omit None values from true_stage and ad_ratios
    filtered_indices = [i for i in range(len(true_stage)) if true_stage[i] is not None and not math.isnan(ad_ratios[i])]
    print(f"Omitted {len(true_stage) - len(filtered_indices)} epochs with None or NaN values.")
    true_stage = [true_stage[i] for i in filtered_indices]
    ad_ratios = [ad_ratios[i] for i in filtered_indices]

    # binarize true_stage: sleep (N1, N2, N3, REM) = 1, wake = 0
    binary_true_stage = [1 if (stage.lower() == 'wake' or stage.lower() == 'w') else 0 for stage in true_stage]

    fpr, tpr, thresholds = roc_curve(binary_true_stage, ad_ratios)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Average Alpha/Delta Ratio - {label}')
    plt.legend(loc="lower right")
    
    # get best threshold
    youden_index = tpr - fpr
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]
    print(f'Best threshold for {label}: {best_threshold:.4f} (TPR = {tpr[best_threshold_index]:.2f}, FPR = {fpr[best_threshold_index]:.2f})')

    # save figure
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "figures", "roc_curves")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "figures", "roc_curves"), exist_ok=True)
    filename = f'ad_roc_curve_{label.replace(" ", "_").lower()}_{current_time}.png'
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "roc_curves", filename))
    print(f"Saved ROC curve for {label} as {filename}.")
    plt.close()

    # plot average alpha delta ratios with best threshold line
    plt.figure()
    plt.hist([ad_ratios[i] for i in range(len(ad_ratios)) if binary_true_stage[i] == 0], bins=30, alpha=0.5, label='Sleep', color='blue')
    plt.hist([ad_ratios[i] for i in range(len(ad_ratios)) if binary_true_stage[i] == 1], bins=30, alpha=0.5, label='Wake', color='orange')
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.2f} (TPR = {tpr[best_threshold_index]:.2f}, FPR = {fpr[best_threshold_index]:.2f})')
    plt.xlabel('Average Alpha/Delta Ratio')
    plt.ylabel('Count')
    plt.title(f'Histogram of Average Alpha/Delta Ratios - {label}')
    plt.legend()
    filename = f'ad_histogram_{label.replace(" ", "_").lower()}_{current_time}.png'
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "roc_curves", filename))
    print(f"Saved alpha/delta ratio histogram for {label} as {filename}.")
    plt.close()

print(f"Total processing time = {timedelta(seconds=total_processing_time)}")