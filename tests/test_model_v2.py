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
        'inf_fraction_param': 2.0,
        'inf_lvl_error_term': -7.0,
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0
    }
    period_length = 60
    n_periods = 24
    simulation_days = 60

    # =========================== ABM SETUP ======================= #
    activity_data = load_period_activities()
    abm = ABM(params, activity_data)

    # ==================== FORCED SIMULATION ====================== #
    forced_infections = [10, 50, 100, 150, 200] + [0] * 20
    abm.force_simulation_start(forced_infections)

    # ========================== SIMULATION ======================= #
    n_days = 20
    abm.run_simulation(n_days, verbose=True)

    # ========================== VISU ============================= #
    new_infections = abm.results.get_per_period("new infections")
    results_df = abm.results.get_per_period_results()

    plt.Figure()
    plt.plot(results_df.index, results_df['infected agents'])
    plt.plot(results_df.index, results_df['positive tests'])
    plt.legend(['infected agents', 'positive tests'])
    plt.show()
