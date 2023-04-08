"""
Tests the ParallelABM class.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abm.parallel import ParallelABM
from abm.contacts import load_period_activities
from abm.plotting import ABM_Plotter

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
    activity_data = load_period_activities()
    rng = np.random.default_rng(seed=42)
    models = ParallelABM(params, activity_data, n_models=4)

    # ========================== SIMULATION ======================= #
    forced_infections = [10, 50, 75, 100, 125, 150, 200]
    results = models.run_simulations(forced_infections, simulation_days)
    results_df = models.get_results_dataframe(timestep='daily')

    # ========================== VISU ============================= #
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(data=results_df, x="day", y="daily summed new infections", ax=ax)
    fig.show()
    print("Done")