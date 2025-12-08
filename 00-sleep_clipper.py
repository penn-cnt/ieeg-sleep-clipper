# 00-sleep_clipper.py

### Parameters to edit ###

# BIDS root
bids_root = r"/mnt/leif/littlab/users/ianzyong/Michigan_Epilepsy_Data/BIDS/"

# BIDS path information
subject = "umich0022"
session = "ieeg01"
datatype = "ieeg"
task = "all"
run = "09"
suffix = "ieeg"
extension = ".lay"

# list of vigilance states to plot, e.g. ['wake', 'N1', 'N2', 'N3', 'REM']
states_to_plot = ['wake', 'N1', 'N2', 'N3', 'REM']

# specifies event number to plot. If None, will plot a random event of that state
event_num = None

# length of time window to plot from event onset (seconds)
window_length = 15

# length of time to plot before onset (seconds)
pre_onset = 0

### End of parameters ###

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mne
from datetime import datetime
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

# set either "qt" or "matplotlib" backend
mne.viz.set_browser_backend("matplotlib")

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

raw = mne.io.read_raw_persyst(bids_path)

for annot in raw.annotations:
    print(f"{annot['onset']}, {annot['duration']}, {annot['description']}")
    continue

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

# if run_events is empty, use comments from the lay file as run_events
if len(run_events) == 0:
    print("No vigilance events found in events.tsv for this run. Using comments from layout file as vigilance events.")
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

for vigilance_state in states_to_plot:
    raw = mne.io.read_raw_persyst(bids_path)
    if run_events.shape[1] == 2:
        # map vigilance state to corresponding code
        state_map = {
            'wake': 'W',
            'N1': '1',
            'N2': '2',
            'N3': '3',
            'REM': 'REM'
        }
        state_events = run_events[run_events[:,1] == state_map[vigilance_state]]
    else:
        # get all events of that vigilance state
        state_events = run_events[run_events[:,3] == vigilance_state]
    print(f"Found {len(state_events)} events for vigilance state {vigilance_state}:")
    print(state_events)
    if len(state_events) == 0:
        print(f"No events found for vigilance state {vigilance_state}")
        continue
    # select random event from state_events
    event_ind = np.random.randint(len(state_events))
    if event_num is not None:
        event_ind = event_num-1
    event_to_plot = state_events[event_ind]
    # get onset time of random event
    # if the first entry contains "T"
    if "T" not in event_to_plot[0]:
        # assume this event came from lay file comments
        onset = round(float(event_to_plot[0]))
    else:
        onset = round(date_to_seconds(event_to_plot[0].split("T")[0]) + time_to_seconds(event_to_plot[0].split("T")[1]) - time_to_seconds(test_time))
    print("Plotted event:", event_to_plot, f"onset time (s): {onset}")
    # load raw data within window_length seconds of that event
    raw.crop(onset - pre_onset, onset + window_length)
    raw.load_data()
    # apply high pass filter at 1 Hz
    raw.filter(1, None)
    # apply low pass filter at 70 Hz
    #raw.filter(None, 70)
    # apply bandstop filter between 58-62 Hz
    raw.notch_filter(60, notch_widths=4)
    # set bipolar montage
    anodes = ["FP1","F7","T3","T5","FP2","F8","T4","T6","FP1","F3","C3","P3","FP2","F4","C4","P4","FZ","CZ"]
    cathodes = ["F7","T3","T5","O1","F8","T4","T6","O2","F3","C3","P3","O1","F4","C4","P4","O2","CZ","PZ"]
    raw = mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes)
    # plot bipolar EEG channels + EKG channels
    ch_names = [anode + "-" + cathode for anode, cathode in zip(anodes, cathodes)] + ["EKG1", "EKG2"]
    # create blank channels for spacing
    blank_ch_names = ["Blank1", "Blank2", "Blank3", "Blank4", "Blank5"]
    raw = raw.add_reference_channels(blank_ch_names)
    # mark as bad
    raw.info['bads'] += blank_ch_names
    # insert blank channels at appropriate locations
    ch_names.insert(18, "Blank5")
    ch_names.insert(16, "Blank4")
    ch_names.insert(12, "Blank3")
    ch_names.insert(8, "Blank2")
    ch_names.insert(4, "Blank1")
    raw.pick(ch_names)
    # plot window_length seconds of data from that event
    raw.plot(n_channels=50, title=f"Vigilance state: {vigilance_state}, event ({event_ind+1}/{len(state_events)}) of this state", duration = window_length, scalings = dict(eeg=50e-6) ,remove_dc=True, splash=False, block=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures", "clips"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "clips", f"{subject}_{run}_{vigilance_state}_interval_{event_ind}_onset_{onset}.png"))
    print("Saved figure to clips folder.")
