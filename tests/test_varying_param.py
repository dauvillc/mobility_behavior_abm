"""
Clément Dauvilliers - EPFL TRANSP-OR Lab 28/02/2023
Tests the effect of varying a single parameter in the ABM while keeping the others
fixed.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abm.parallel import ParallelABM

if __name__ == "__main__":
    # ====================================== INIT ================= #

    # Simulation parameters
    params = {
        'inf_params': {'age': 0.000},
        'test_params': {'age': 0.000},
        'inf_fraction_param': 35,
        'inf_lvl_error_term': -30.0,
        'inf_proba_sigmoid_slope': 0.2,
        'test_inf_lvl_param': 1.0,
        'test_error_term': -13,
        'test_proba_sigmoid_slope': 1.0,
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0
    }
    period_length = 60
    n_periods = 24
    simulation_days = 30

    # =========================== ABM SETUP ======================= #
    rng = np.random.default_rng(seed=42)
    models = ParallelABM(params, n_models=4)
    models.seeds = [42] * models.n_models

    param_values = [0, 15, 30, 35]
    models.set_varying_param("inf_fraction_param", param_values)

    # ========================== SIMULATION ======================= #
    forced_infections = [10, 50, 75, 100, 125, 150, 200]
    models.run_simulations(forced_infections, simulation_days)
    results_df = models.get_results_dataframe(timestep='daily')

    # ========================== VISU ============================= #
    # Creates a column "inf_fraction_param" in the results dataframe, so that it
    # will be automatically set as the legend by seaborn
    simulation_params = pd.Series([str(val) for val in param_values], name="inf_fraction_param")
    results_df = results_df.merge(simulation_params, left_on="simulation", right_index=True)

    sns.set_theme()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
    fig.suptitle('Epidemic trajectories with varying force of infection')
    sns.lineplot(data=results_df, x="day", y="daily summed new infections", hue="inf_fraction_param",
                 style="inf_fraction_param",
                 ax=ax, palette=sns.color_palette('icefire', n_colors=len(param_values)))
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily new infections")
    plt.savefig('tests/figures/varying_param.png')
    fig.show()
    print("Done")
