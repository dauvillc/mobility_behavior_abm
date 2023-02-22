"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 31/01/2023
Tests the forced simulation methods of the ABM.
"""
import numpy as np
from abm.model import ABM
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
    simulation_days = 60

    # =========================== ABM SETUP ======================= #
    activity_data = load_period_activities()
    abm = ABM(params, activity_data)

    # ==================== FORCED SIMULATION ====================== #
    forced_daily_inf = [10, 50, 100, 150, 200]
    abm.force_simulation_start(forced_daily_inf)

    # ========================== SIMULATION ======================= #
    abm.simulation(simulation_days, verbose=True)

    # ========================== VISU ============================= #
    plotter = ABM_Plotter(abm)
    plotter.plot_curves(save_img_to="tests/figures/simulation.jpg", show_fig=True)
    # plotter.plot_infection_spread(save_html_to="tests/figures/geo_spread.html", show_fig=True)
    print("Done")