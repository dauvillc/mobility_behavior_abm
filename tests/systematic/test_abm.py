"""
Clément Dauvilliers - April 2nd 2023
Tests the ABM class.
"""
import numpy as np
import pandas as pd
from abm.contacts import load_period_activities
from abm.model_v3 import ABM

if __name__ == "__main__":
    # Loads the population socio-eco attributes
    print("Loading the population dataset..")
    population_df = pd.read_csv('data/abm/vaud/extracted/vaud_population.csv.gz', index_col='agent_index')
    population_df = population_df.sort_index()
    print("Done")
    activity_data = load_period_activities()

    # Builds the agents' characteristics
    params = {
        'inf_params': {'age': 0.000},
        'test_params': {'age': 0.000},
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0
    }
    # Builds the ABM object
    abm = ABM(params, activity_data)

    # ACTUAL TEST
    # Tests the forced simulation start
    forced_infections = [100, 200, 300, 400, 500]
    abm.force_simulation_start(forced_infections)
    # Some basic verifications regarding the length and values of simulation variables
    assert abm.day == len(forced_infections)  # abm.day starts at zero
    assert abm.period == 0
    new_infections_per_period = abm.results.get_per_period("new infections")
    assert len(new_infections_per_period) == len(forced_infections) * abm.n_periods
    # The following lines verify that the number of infections that occurred is roughly
    # equal to its expected value.
    total_infections = sum(new_infections_per_period)
    expected_infections = sum(forced_infections)
    assert expected_infections * 0.9 <= total_infections <= expected_infections * 1.1


    print("Test successful !")