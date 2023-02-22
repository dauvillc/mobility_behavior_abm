"""
Clément Dauvilliers - 26/09/2022 - EPFL Transport and Mobility lab

For every facility, counts the number the people of every age who visited it during the day.
Uses as input a list of facilities visited by the population for every period of a single day, as output by
preprocess_matsim_data.py.
"""

import sys

import pandas as pd
import os
import csv
from optparse import OptionParser
from collections import defaultdict
from tqdm import tqdm

# PATHS
# Preprocessed activities file
ACTIVITIES_FILE = "data/preprocessed/preprocessed_activities.csv.gz"
# Directory to store the results
DEST_DIR = "data/preprocessed/facilities_age_distrib/"
# Chunksize for reading the file (Reduce that value to decrease the RAM usage)
# (Number of CSV rows loaded in memory at once)
CHUNKSIZE = 500000

if __name__ == "__main__":
    # PARSING OPTIONS
    optparser = OptionParser()
    optparser.add_option('--period_length', action="store_const", dest="period_length", default="H",
                         help="pandas string period indicator, such as 'H' for an hour or '30T' for 30 minutes. "
                              "Indicates the length of the time periods when approximating the activities' starting"
                              "and ending times. For example, 'H' will generate to periods such as '09:00' to '10:00'.")
    (options, args) = optparser.parse_args()

    print(f"Beginning preprocessing with a chunksize of {CHUNKSIZE}...")

    # HOWTO
    # We run the program once for each period of the day, to limit the RAM requirement.
    # We use dictionaries to count the attendance of facilities:
    # facilities_map is a hashmap (Activity type) --> age_counts
    # age_counts is a hashmap (Age) --> count

    # Recreates the list of all periods based on the period length
    periods = pd.date_range("1900-01-01 00:00:00",
                            "1900-01-01 23:59:59",
                            freq=options.period_length)

    # We process period by period to limit the needed RAM
    for index, current_period in enumerate(periods):
        print("Processing period ", current_period)

        facilities_map = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # We process the activities CSV once fully for every period. We process it in chunks as it would
        # be too large to fit in memory.
        with pd.read_csv(ACTIVITIES_FILE, chunksize=CHUNKSIZE, parse_dates=['period']) as full_csv:
            for it, chunk in tqdm(enumerate(full_csv), total=404):
                # Only keeps the lines that concern the period that is being processed
                chunk = chunk[chunk['period'] == current_period]
                # For every row of the filtered chunk, adds its data to the facilities map
                for _, id, type, facility, age, period in chunk.itertuples():
                    facilities_map[facility][type][age] += 1

            nb_facilities = len(facilities_map.keys())
            print(f"Found {nb_facilities} unique facilities")

        dest_file = os.path.join(DEST_DIR,
                                 current_period.strftime('%H-%M') + ".csv")
        with open(dest_file, "w", newline='') as dest_file:
            fieldnames = ['facility', 'period', 'type', 'age', 'count']
            writer = csv.DictWriter(dest_file, fieldnames=fieldnames)

            print("Writing to ", dest_file)
            writer.writeheader()
            period_str = current_period.strftime('%H:%M:%S')
            for facility in tqdm(facilities_map.keys(), total=nb_facilities):
                for type in facilities_map[facility].keys():
                    for age in facilities_map[facility][type].keys():
                        writer.writerow({'facility': facility,
                                         'period': period_str,
                                         'type': type,
                                         'age': age,
                                         'count': facilities_map[facility][type][age]})
