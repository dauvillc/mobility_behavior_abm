"""
Clément Dauvilliers - April 2nd 2023
Tests the new version of the ABM.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abm.model_v3 import ABM
from abm.contacts import load_period_activities

if __name__ == "__main__":
    # ====================================== INIT ================= #

    # Simulation parameters
    params = {
        'inf_params': {'age': 0.000},
        'test_params': {'age': 0.000},
        'inf_fraction_param': 16.50,
        'inf_lvl_error_term': -11.0,
        'test_inf_lvl_param': 2.0,
        'test_error_term': 0,
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0
    }
    period_length = 60
    n_periods = 24

    # =========================== ABM SETUP ======================= #
    activity_data = load_period_activities()
    abm = ABM(params, activity_data)

    # ==================== FORCED SIMULATION ====================== #
    forced_infections = [10, 50, 100, 150, 200]
    abm.force_simulation_start(forced_infections)

    # ========================== SIMULATION ======================= #
    simulation_days = 60
    abm.run_simulation(simulation_days, verbose=True)

    # ========================== VISU ============================= #
    new_infections = abm.results.get_per_period("new infections")
    results_df = abm.results.get_per_period_results()

    plt.Figure()
    plt.plot(results_df.index, results_df['new infections'])
    plt.legend(['new infections'])
    plt.show()
