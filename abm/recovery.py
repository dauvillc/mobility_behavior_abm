"""
Clément Dauvilliers - April 2nd 2023
Implements the methods related to the recovery process.
"""
from abm.utils import compute_lognormal_params


def draw_recovery_times(n_agents, mean_recov_time, std_recov_time,
                        n_periods, rng):
    """
    For a given number of agents, randomly draws their recovery times from a
    lognormal distribution.
    Parameters
    ----------
    n_agents: integer, number of times to draw.
    mean_recov_time: int, mean recovery time in days.
    std_recov_time: int, std of the recovery time in days.
    n_periods: int, number of periods per day.
    rng: numpy random generator to use.
    """
    # Compute the mean and std of the underlying normal distrib
    mean_n, std_n = compute_lognormal_params(mean_recov_time, std_recov_time)
    times = rng.lognormal(mean=mean_n, sigma=std_n, size=n_agents)
    # The times were drawn in days, but we need to convert them to periods
    times *= n_periods
    return times
