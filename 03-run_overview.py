# 03-run_overview.py
# automatically detects all runs in the data directory and outputs graphs showing run durations for each patient and discontinuities between runs
# USAGE: python 03-run_overview.py

import os
import matplotlib.pyplot as plt
import numpy as np
import time
import mne
import math
from datetime import datetime
from datetime import timedelta
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
    write_raw_bids
)
import pandas as pd
import sys
from dateutil import parser

def onset_str_to_delta_seconds(start_onset_str, stop_onset_str):
    # gets delta seconds between two onset strings
    start_time = parser.parse(start_onset_str)
    stop_time = parser.parse(stop_onset_str)
    delta = stop_time - start_time
    return delta.total_seconds()

#bids_root = os.path.join("/mnt/leif/littlab/users/ianzyong/Michigan_Epilepsy_Data/BIDS/")
bids_root = r"C:\Users\ianzy\Documents\Research\sleep-benchmark\Michigan_Epilepsy_Data\BIDS"
# output path for plots
os.makedirs(os.path.join(os.path.dirname(__file__), 'figures', 'plots'), exist_ok=True)
# get all subjects
all_subjects = get_entity_vals(bids_root, 'subject')
print("Total number of subjects:", len(all_subjects))
print("Reading run information from events.tsv files...")
for subject in all_subjects:
    print(f"\n=== Subject {subject} ===")
    # BIDS path
    bids_path = BIDSPath(
        subject=subject,
        root=bids_root,
        session="ieeg01",
        datatype="ieeg"
    )
    # read events.tsv to get run information
    events_tsv_path = bids_path.copy().update(suffix="events", extension=".tsv", task=None, run=None)
    # read tsv as dataframe
    events_df = pd.read_csv(events_tsv_path.fpath, sep='\t')
    # filter for vigilance events
    vigilance_events_df = events_df[events_df['event_category'] == 'vigilance']
    # output run durations where durations with unknown vigilance represents a discontinuity
    # find all rows where vigilance is unknown
    unknown_vigilance_df = vigilance_events_df.loc[vigilance_events_df['event_type'] == 'unknown']
    # reset indices
    unknown_vigilance_df = unknown_vigilance_df.reset_index(drop=True)
    # assume that the events file starts and ends with unknown vigilance events
    first_onset_str = vigilance_events_df.iloc[1]['onset']
    print("Start time of first run (24-hr format):", first_onset_str.split('T')[1])
    print(f"Total number of runs: {len(unknown_vigilance_df) - 1}")
    print("Run durations:")
    run_durations = []
    for index, row in unknown_vigilance_df.iterrows():
        # if last index
        if index == unknown_vigilance_df.index[-1]:
            continue
        # print duration between this unknown vigilance event and the next unknown vigilance event
        onset_str = row['onset']
        # get index of next row
        next_onset_str = unknown_vigilance_df.loc[index+1]['onset']
        delta_seconds = onset_str_to_delta_seconds(onset_str, next_onset_str)
        # subtract duration of the first unknown vigilance event
        delta_seconds -= row['duration']
        run_durations.append(delta_seconds)
        print(f"\tRun {index+1}: {str(timedelta(seconds=delta_seconds))}")
        next_unknown_duration = unknown_vigilance_df.loc[index+1]['duration']
        if not math.isnan(next_unknown_duration):
            print(f"\t(Unknown vigilance: {str(timedelta(seconds=next_unknown_duration))})")
    print(f"Total duration of runs: {str(timedelta(seconds=sum(run_durations)))}")
