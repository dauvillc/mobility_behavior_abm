"""
Clément Dauvilliers - April 2nd 2023
Tests the ABM class.
"""
import numpy as np
import pandas as pd
from abm import ABM

if __name__ == "__main__":
    # Creates an RNG just in case we need one:
    rng = np.random.default_rng(seed=42)

    # Simulation parameters
    params = {
        'inf_params': {'age': 0.000},
        'test_params': {'age': 0.000},
        'inf_fraction_param': 13,
        'inf_lvl_error_term': -11.0,
        'inf_proba_sigmoid_slope': 2.0,
        'test_inf_lvl_param': 1.0,
        'test_error_term': -2,
        'test_proba_sigmoid_slope': 0.5,
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0
    }
    # Builds the ABM object
    abm = ABM(params)

    # ACTUAL TEST
    # Tests the forced simulation start ========================================
    forced_infections = [100, 200, 300, 400, 500]  # dummy values
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
    # Tests the actual simulation ==============================================
    n_days = 20
    abm.run_simulation(n_days, verbose=True)
    # Same basic verifications as with the forced start
    assert abm.day == len(forced_infections) + n_days
    assert abm.period == 0
    new_infections_per_period = abm.results.get_per_period("new infections")
    assert len(new_infections_per_period) == (len(forced_infections) + n_days) * abm.n_periods
    # Verifies that the number of infected visitors is always inferior or equal
    # to the number of visitors:
    visitors = abm.population.mobility.get_visitors(0)
    infected_visitors = abm.population.mobility.get_infected_visitors(0)
    assert np.all(infected_visitors <= visitors)
    # Verifies that the number of infected visitors in total is equal to the number
    # of infected agents
    assert infected_visitors.sum() == abm.population.get_state_count("infected")
    assert len(abm.population.get_infected_agents()) == infected_visitors.sum()

    # Tests that the ABM resets well
    forced_infections = [100] + [0] * 50  # Includes 50 days without any infection
    abm.force_simulation_start(forced_infections)
    new_infections_per_period = abm.results.get_per_period("new infections")
    total_infections = sum(new_infections_per_period)
    expected_infections = sum(forced_infections)
    assert expected_infections * 0.9 <= total_infections <= expected_infections * 1.1
    # The following checks that there are no infected agents in the last days
    infected_agents_count = abm.results.get_per_period("infected agents")
    assert sum(infected_agents_count[-10:]) == 0

    # Tests the activity reduction
    forced_infections = [10, 20, 30, 40]
    abm.force_simulation_start(forced_infections)
    day, period = abm.day, abm.period  # current simulation date
    some_agents = rng.integers(0, abm.n_agents, 15)  # Selects some random agents
    # Run the simulation for fewer days than the duration of the reduction
    duration_days, duration_periods = 5, 3
    abm.reduce_mobility(some_agents, duration_days, duration_periods)
    abm.run_simulation(duration_days - 1)
    # Verifies that the agents are indeed confined
    assert (abm.population.mobility.locations[0][some_agents] == 0).all()
    # Runs the simulation until after the reduction is lifted
    abm.run_simulation(2)
    # Verifies that the agents aren't confined anymore, i.e. that their locations
    # are equal to the original ones
    assert (abm.population.mobility.locations[0][some_agents] == abm.population.mobility.original_locations[0][
        some_agents]).all()

    print("Test successful !")
