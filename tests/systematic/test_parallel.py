"""
Clément Dauvilliers - April 8th 2023
Tests the ParallelABM class.
"""
import numpy as np
from abm.parallel import ParallelABM
from abm.contacts import load_period_activities

if __name__ == "__main__":
    # ====================================== INIT ================= #

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
    period_length = 60
    n_periods = 24
    simulation_days = 14

    # =========================== ABM SETUP ======================= #
    rng = np.random.default_rng(seed=42)
    models = ParallelABM(params, n_models=4)

    # ========================== SIMULATION ======================= #
    # Performs a first batch of parallel simulations
    forced_infections = [10, 50, 75, 100, 125, 150, 200]
    models.run_simulations(forced_infections, simulation_days)
    results_df = models.get_results_dataframe(timestep='daily')

    # Basic tests regarding the results
    assert results_df['day'].max() + 1 == len(forced_infections) + simulation_days

    # Resets the ParallelABM object, and performs the same simulations
    models.reset()
    models.run_simulations(forced_infections, simulation_days)
    results_second_time = models.get_results_dataframe(timestep='daily')

    # Checks that the results are exactly the same as before
    assert results_second_time.equals(results_df)

    print("Test successful !")