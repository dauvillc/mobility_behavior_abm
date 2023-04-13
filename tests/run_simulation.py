"""
Clément Dauvilliers - April 2nd 2023
Tests the new version of the ABM.
"""
import numpy as np
from abm.model import ABM
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

    # =========================== ABM SETUP ======================= #
    abm = ABM(params)

    # ==================== FORCED SIMULATION ====================== #
    forced_infections = [10, 50, 75, 100, 125, 150, 200]
    abm.force_simulation_start(forced_infections)

    # ========================== SIMULATION ======================= #
    simulation_days = 30
    abm.run_simulation(simulation_days, verbose=True)

    # ========================== VISU ============================= #
    plotter = ABM_Plotter(abm)
    plotter.plot_curves(save_img_to="tests/figures/simulation.jpg", show_fig=True)
    # plotter.plot_infection_spread(save_html_to="tests/figures/geo_spread.html", show_fig=True)
