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
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0,
        'inf_params': {'age': 0.001},
        'inf_fraction_param': 0.045,
        'test_params': {'age': 0.001},
        'base_test_proba': 0.1,
        'inf_test_proba': 0.5,
        'apply_activity_reduction': False
    }
    period_length = 60
    n_periods = 24
    simulation_days = 14

    # =========================== ABM SETUP ======================= #
    activity_data = load_period_activities()
    rng = np.random.default_rng(seed=42)
    models = ParallelABM(params, activity_data, n_models=4)
    initial_infections = rng.random(models.models[0].n_agents) < 0.001

    # ========================== SIMULATION ======================= #
    models.init_simulation(initial_infections)
    results = models.run_simulations(simulation_days)
    results_df = models.get_results_dataframe(timestep='daily')

    # ========================== VISU ============================= #
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(data=results_df, x="day", y="total tests", ax=ax)
    sns.lineplot(data=results_df, x="day", y="positive tests", ax=ax)
    fig.show()
    print("Done")