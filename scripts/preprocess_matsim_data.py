"""
Clément DAUVILLIERS - 20/09/2022 - EPFL Transport and Mobility Lab

Usage: python3 scripts/preprocess_matsim_data.py [--help] [--age_groups]
Parses the MATSIM activities extracted from the output XML plans file. Adds the age
of every individual alongside the activities. Processes the activities' starting and ending times
to turn them into periods (e.g., from 09:00:00 to 11:00:00).

Use the --help option to see the possible options.
"""
import sys

import pandas as pd
import os
from optparse import OptionParser
from tqdm import tqdm

# PATHS
# Path to the whole activities csv extracted from the MATSIM XML results file (plans)
MATSIM_ACTIVITIES_PATH = "data/extracted_data/matsim_population_activities.csv"
# Path to the whole individual attributes csv extracted from the XML file too
MATSIM_ATTRIBUTES_PATH = "data/extracted_data/matsim_population_attributes.csv"
# Destination file
DEST_FILE = "data/preprocessed/preprocessed_activities.csv.gz"

if __name__ == "__main__":
    # PARSING OPTIONS
    optparser = OptionParser()
    optparser.add_option('--chunksize', action="store", dest="chunksize", type="int", default=1000000,
                         help="Number of CSV rows to process at once (Default: 1,000,000). A "
                              "larger value will lead to a shorter processing time, but requires more"
                              " memory.")
    optparser.add_option("--age_groups", action="store_true", dest="age_groups", default=False,
                         help="Transform the age into age groups [default: don't]")
    optparser.add_option("--explode_periods", action='store_true', dest='explode_periods',
                         default=True,
                         help="If False (default True), a single row will be output for each activity, indicating its "
                              "starting and ending times. If True, a row will be output for every activity and for "
                              "every period. For example, an activity from 09:13 to 10:50 will be written in two "
                              "rows, one from 09:00 to 10:00 and another from 10:00 to 11:00. If activated, "
                              "the columns 'start_time' and 'end_time' will be replaced by a single column 'period'.\n"
                              "BE CAREFUL, as this option will dramatically increase the size of the output file, "
                              "especially with short time periods such as 1H or 30min.")
    optparser.add_option('--period_length', action="store_const", dest="period_length", default="H",
                         help="pandas string period indicator, such as 'H' for an hour or '30T' for 30 minutes. "
                              "Indicates the length of the time periods when approximating the activities' starting"
                              "and ending times. For example, 'H' will generate to periods such as '09:00' to '10:00'.")
    (options, args) = optparser.parse_args()

    print(f"Beginning preprocessing with a chunksize of {options.chunksize}...")
    # We'll write in the results file in 'append' mode, hence not overwriting the file
    # if it already exists.
    if os.path.exists(DEST_FILE):
        print("The destination file ", DEST_FILE, " already exists.")
        sys.exit(0)

    # We've got the age of every individual (i.e. id) in the attributes file.
    # Nonetheless, we won't use the raw age but the age group.
    population_data = pd.read_csv(MATSIM_ATTRIBUTES_PATH, usecols=['id', 'age'])
    # Fetches the id-age association
    id_age_df = population_data[['id', 'age']].drop_duplicates().set_index('id')
    if options.age_groups:
        # Creates the age groups
        age_intervals = pd.DataFrame({'From': [0, 10, 20, 36, 51, 65], 'To': [9, 19, 35, 50, 64, 150],
                                      'Group': ['0-9', '10-19', '20-35', '36-50', '51-64', '65-']})
        # Converts to pandas IntervalIndex objects which will automatically be mapped to the age column
        age_intervals = age_intervals.set_index(pd.IntervalIndex.from_arrays(age_intervals['From'],
                                                                             age_intervals['To'], closed="both"))[
            'Group']

    # We process the CSV chunkwise as it is usually too large to fit in memory
    # The "header" variable will make pandas write the DF's header row, but only the first time
    write_header = True
    with pd.read_csv(MATSIM_ACTIVITIES_PATH, index_col=0, chunksize=options.chunksize) as full_csv:
        for it, activities_df in tqdm(enumerate(full_csv), total=42000000/options.chunksize):
            # We'll need to remove the rows in which the facility is null
            activities_df = activities_df[activities_df['facility'].notna()]
            # Removes unused columns
            activities_df = activities_df.drop(['link', 'x', 'y'], axis=1)

            # ADDING THE AGE
            # The following adds the raw age to the dataframe
            activities_df = activities_df.merge(id_age_df, left_on="id", right_index=True)
            if options.age_groups:
                activities_df['age_group'] = activities_df['age'].map(age_intervals)
                # We won't need the raw age any longer.
                activities_df = activities_df.drop('age', axis=1)

            # PREPROCESSING THE ACTIVITY TIME
            # Some activities do not have a starting nor an ending time (NaN), to indicate that they haven't started
            # or ended at any point during the day. We'll replace these NaNs with "00:00" for the starting time and
            # 23:59 for the ending time. We'll also discard the seconds.
            activities_df['start_time'] = activities_df['start_time'].fillna('00:00:00')
            activities_df['end_time'] = activities_df['end_time'].fillna('23:59:59')
            # Converts the times from strings to datetime format
            activities_df['start_time'] = pd.to_datetime(activities_df['start_time'], format="%H:%M:%S",
                                                         errors="coerce")
            activities_df['end_time'] = pd.to_datetime(activities_df['end_time'], format="%H:%M:%S", errors="coerce")
            # Some start / end times are beyond 24h. They were converted to "NaT".
            # starting times over 24:00 mean the activity started on the next day, which we won't consider:
            activities_df = activities_df[~(activities_df['start_time'].isnull())]
            # ending times over 24:00 mean the activity ended on the next day, but if it's still here it means
            # it started before 24:00. So we'll keep the activity but set the end time to 23:59:
            activities_df['end_time'] = activities_df['end_time'].fillna(
                pd.Timestamp(year=1900, month=1, day=1, hour=23,
                             minute=59, second=59))
            # Finally, we'll transform the raw, precise times (HH:MM) to periods. For example, 09:54 becomes
            # 09:00 if a period of 1 hour is used. 09:00 indicates the period 09:00 to 10:00.
            # Use 'H' for an hour or '30T' for 30 minutes.
            activities_df['start_time'] = activities_df['start_time'].dt.floor(options.period_length)
            activities_df['end_time'] = activities_df['end_time'].dt.floor(options.period_length)

            if options.explode_periods:
                # Explodes the dataframe based on the time period. For example an activity starting at 10:00 and
                # ending 10:30, with a period of 10 min, will result in 3 identical rows (one for 10:00, for 10:10,
                # and 10:20). The resulting dataframe can be MUCH larger if the time period is especially short.

                # The following creates a columns whose values are lists of periods from start_time to end_time
                activities_df['period'] = activities_df.apply(
                    lambda row: pd.date_range(row['start_time'], row['end_time'], freq=options.period_length), axis=1)
                # We can now drop the start_time and end_times columns (and we should to avoid multiplying the rows
                # after exploding !)
                activities_df = activities_df.drop(['start_time', 'end_time'], axis=1)
                # This explodes the dataset with regard to the lists just created
                activities_df = activities_df.explode('period')
                activities_df.head(10)

            # Saves the preprocessed data
            activities_df.to_csv(DEST_FILE, mode="a", header=write_header)
            # After we wrote once into the results file, do not write the header any once more.
            write_header = False
