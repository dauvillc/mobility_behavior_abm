"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 27/11/2022
Defines functions to load and manage contact data.
"""
import json
import gzip
import numpy as np
import os
import pandas as pd
import scipy.sparse as sparse


def load_agents_visits():
    """
    Loads the visits list A such that
    A[i] is the list of the facilities that agent i
    visited in the simulation.
    """
    path = "data/abm/vaud/prepared/vaud_agents_visits.json.gz"
    with gzip.open(path, 'r') as fin:
        return json.loads(fin.read().decode('utf-8'))


def load_facilities_visitors():
    """
    Loads the list F such that
    F[f] is the list of the agents who visited facility f
    during the simulation.
    """
    path = "data/abm/vaud/prepared/vaud_facilities_visitors.json.gz"
    with gzip.open(path, 'r') as fin:
        return json.loads(fin.read().decode('utf-8'))


def load_period_activities(period_activities_dir="data/abm/vaud/prepared/period_activities/"):
    """
    Loads the activities per period, and builds the associated visit
    and visitors sparse matrices.
    Parameters
    ----------
    period_activities_dir: str, optional. Path to the directory in which the period activities are stored, with
    a single CSV file for each period. The default value is the default location of those files after
    they are created by the prepare_data.ipynb notebook.
    Returns
    -------
    A triplet LV, LA, LT:
    -- LV is a list of length N_p sparse matrices, with Np being the number of periods.
    Each matrix is a matrix V such that V[f, i] is 1 if agent i visited f
    during the period, and 0 otherwise. The matrices are stored in the COO format.
    -- LA is a list of length N_p arrays, each of shape (n_agents). For a period p, the array LA[p]
    is such that LA[p][i] is the index of the facility that agent i was at during period
    p. If the agent visited several facilities during p, the last one is kept.
    -- LT is a list of length N_p arrays of shape (n_agents). For a period p, the array LT[p]
    is such that LT[p][i] is the type (as an str) of the activity that agent i performed during the
    period p.
    """
    print("Creating activity matrices...")
    # The first will contain sparse matrices of shape (n_agents, n_facilities)
    # The second will contain arrays of shape (n_agents)
    visit_matrices, agent_locations, act_types = [], [], []
    for period, act_file in enumerate(os.listdir(period_activities_dir)):
        print(f"Processing period {period}")
        # Read the activities file for the given period
        activities = pd.read_csv(os.path.join(period_activities_dir, act_file))
        # If an agent performed several activities, we only keep the first one
        activities.drop_duplicates(keep="first", inplace=True)
        # Creates the sparse visit matrix
        data = np.full(activities.shape[0], 1)
        visit_matrices.append(sparse.coo_array(
            (data, (activities['facility_index'], activities['agent_index']))
        ))
        # Creates the agent visits arrays. We use the value 0 to indicate
        # that the location of the agent is unknown during that period.
        agents_period_visits = np.zeros(visit_matrices[0].shape[1], dtype=np.int)
        agents_period_visits[activities['agent_index']] = activities['facility_index']
        agent_locations.append(agents_period_visits)
        # Saves the activity types for that period, and convert it to a numpy array because it uses
        # much less memory than a pandas Series
        period_act_types = activities['type'].to_numpy()
        act_types.append(period_act_types)
    return visit_matrices, agent_locations, act_types
