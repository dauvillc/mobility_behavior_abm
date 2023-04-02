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
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0,
        'inf_params': {'age': 0.000},
        'inf_fraction_param': 0.01375,
        'test_params': {'age': 0.000},
        'base_test_proba': 1.0,
        'test_inf_proba_factor': 0.0,
        'apply_activity_reduction': False
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

    # ========================== VISU ============================= #
    new_infections = abm.results.get_per_period("new infections")
    results_df = abm.results.get_per_period_results()

    plt.Figure()
    sns.lineplot(results_df, x=results_df.index, y="infected agents")
    plt.show()
