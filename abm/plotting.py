"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - December 12th 2022
Implements classes and functions to plot results and information about
the agent-based model.
"""
import io

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import PIL
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ABM_Plotter:
    """
    The ABM_Plotter object is designed to receive results from the
    model only once, and then create multiple figures.
    """

    def __init__(self, abm_object):
        """

        Parameters
        ----------
        abm_object: ABM object whose simulation has already been run.

        Returns
        -------

        """
        self.abm = abm_object
        self.daily_results = self.abm.results.get_daily_results()
        self.n_days = abm_object.day

    def plot_curves(self, save_img_to=None, show_fig=False):
        """
        Plots various curves describing the simulation's results.
        Parameters
        ----------
        save_img_to: str, optional. Path to where the figure should be
            saved.
        show_fig: boolean, optional. Whether to show the figure (default: don't).
        """
        sns.set_theme()

        days = np.array(range(self.abm.day))

        title = "Epidemic incidence over time"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=self.daily_results['daily summed new infections'],
                                 mode="lines+markers", name="daily summed new infections"))
        fig.add_trace(go.Scatter(x=days, y=self.daily_results['daily summed positive tests'],
                                 mode="lines+markers", name="daily summed positive tests"))
        fig.update_layout(title=title, xaxis_title="Day", yaxis_title="Incidence")

        if save_img_to is not None:
            fig.write_image(save_img_to)
        if show_fig:
            fig.show()

    def plot_infection_spread(self, save_html_to=None,
                              save_gif_to=None,
                              show_fig=False):
        """
        Plots the geographical spread of the epidemic over time.

        Parameters
        ----------
        save_html_to: str, optional. Path to an HTML file into which the animation
            shall be saved.
        save_gif_to: str, optional. Path to a GIF image into which the animation
            shall be saved.
        show_fig: boolean, optional. Whether to display the figure.

        Returns
        -------
        a matplotlib Figure object.

        """
        abm = self.abm
        # DataFrame containing the information about each agent
        population_df = abm.population.population_dataset
        # List of the giving the IDs of the infected agents during each period.
        infected_ids = abm.results.get("infected agents IDs")
        # Creates a list of DataFrames corresponding to the successive simulation periods.
        # Each DF contains the index and coordinates of all agents who
        # are infected during the associated period.
        inf_locations = [population_df.loc[indexes, ['wgs84_e', 'wgs84_n', 'municipality']] for indexes in
                         infected_ids]
        # Each dataframe 'period_inf_locations' contains the coordinates of all infected agents
        # during that period.
        period_town_infections = []
        for period, period_inf_locations in enumerate(inf_locations):
            # The following sums the current cases at every town, during
            # the period that is being treated.
            town_infections = period_inf_locations.value_counts().rename(period)
            period_town_infections.append(town_infections)
        # Concatenates into a single dataframe whose index is the locations,
        # and columns are the periods
        period_town_infs_df = pd.concat(period_town_infections, axis=1)
        # If a location did not have any infections during a period, it will create a NaN value.
        # Since those represent "no cases", they're actually a number of cases equal to zero.
        period_town_infs_df = period_town_infs_df.fillna(0)
        # Converts the DataFrame to long format, adapted to Plotly functions
        # The columns are now 'period', and 'infections'; the index is still the locations.
        period_town_infs_df = period_town_infs_df.melt(value_vars=period_town_infs_df.columns,
                                                       var_name='period',
                                                       value_name='infections',
                                                       ignore_index=False).reset_index()
        # Sorts by period, so that we can then compute the cumulative sum over the period in the right
        # order
        period_town_infs_df = period_town_infs_df.sort_values('period')
        # Adds a column 'day', which will for example be 0 for periods 0 to 23.
        period_town_infs_df['day'] = period_town_infs_df['period'] // abm.n_periods
        # Sums the cases over all periods of the same day, to obtain daily data
        daily_local_infs = period_town_infs_df.groupby(['day', 'municipality', 'wgs84_e', 'wgs84_n'])[
            'infections'].sum().reset_index()

        # ===================== FIGURE ==================================== #

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Local density of cases over time",
                                            "Total epidemic incidence over time"),
                            row_heights=[0.7, 0.3], vertical_spacing=0.05,
                            specs=[[{'type': "mapbox"}], [{"type": "xy"}]])

        # Based on the answer from @empet at
        # https://community.plotly.com/t/animation-two-different-plots-using-traces/63984
        # First frame (base figure before the animation starts)
        # Creates the map subplot
        first_period_data = daily_local_infs.query('day == 0')
        zmax = daily_local_infs['infections'].max()
        fig.add_trace(go.Densitymapbox(lat=first_period_data['wgs84_n'],
                                       lon=first_period_data['wgs84_e'],
                                       z=first_period_data["infections"],
                                       colorbar=dict(len=0.6, thickness=30, title="Local infections"),
                                       zmin=1, zmax=zmax),
                      row=1, col=1)

        # Creates the fixed curve subplot
        daily_infections = self.daily_results['daily summed new infections']
        days = np.arange(0, self.n_days)
        curve_ylim = int(max(daily_infections) * 1.1)
        fig.add_scatter(x=days, y=daily_infections,
                        mode="lines+markers", name="Incidence",
                        line=dict(color="royalblue"),
                        row=2, col=1)
        fig.add_scatter(x=[0, 0], y=[0, curve_ylim],
                        mode="lines", line=dict(width=2, dash='dash', color="firebrick"),
                        opacity=0.8,
                        row=2, col=1)

        # Successive frames
        frames_density_data = [daily_local_infs.query(f"day == {d}") for d in range(self.n_days)]
        frames = [
            go.Frame(data=[go.Densitymapbox(lat=frames_density_data[k]['wgs84_n'],
                                            lon=frames_density_data[k]['wgs84_e'],
                                            z=frames_density_data[k]["infections"],
                                            zmin=1, zmax=zmax),
                           go.Scatter(x=[k, k], y=[0, curve_ylim - 1],
                                      mode="lines", line=dict(width=2, dash='dash', color="firebrick"),
                                      opacity=0.8)],
                     name=f"frame{k}", traces=[0, 2])
            for k in range(self.n_days)
        ]
        fig.update(frames=frames)

        # Frames and Slider update
        def frame_args(duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

        # Duration of the frames in milliseconds
        frame_duration_ms = 1000
        # Defines the periods slider
        sliders = [
            {
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(frame_duration_ms)],
                        "label": f"+{k} days",
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]
        # Defines the play / pause buttons
        fig.update_layout(sliders=sliders,
                          updatemenus=[
                              {
                                  "buttons": [
                                      {
                                          "args": [None, frame_args(frame_duration_ms)],
                                          "label": "&#9654;",  # play symbol
                                          "method": "animate",
                                      },
                                      {
                                          "args": [[None], frame_args(frame_duration_ms)],
                                          "label": "&#9724;",  # pause symbol
                                          "method": "animate",
                                      }],
                                  "direction": "left",
                                  "pad": {"r": 10, "t": 70},
                                  "type": "buttons",
                                  "x": 0.1,
                                  "y": 0,
                              }])

        # Mapbox layout update
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_bounds={"west": 6.1, "east": 7.2, "south": 46.2, "north": 46.95}
        )
        # Curve layout update
        fig.update_xaxes(title="Days")
        fig.update_yaxes(range=[0, curve_ylim],
                         title="Number of infected agents (non-tested included)")

        # General figure updates
        fig.update_layout(title="ABM Simulation summary",
                          showlegend=False,
                          margin=dict(t=30))
        if save_html_to is not None:
            plotly.offline.plot(fig, filename=save_html_to, auto_open=False)
        # The method to save the animation as a GIF is taken from
        # https://stackoverflow.com/questions/55460434/how-to-export-save-an-animated-bubble-chart-made-with-plotly
        if save_gif_to is not None:
            print("Saving to GIF..")
            gif_frames = []
            for s, fr in enumerate(fig.frames):
                # Sets the figure as it should be during the sth frame
                data_mapbox, data_cursor = fr.data
                fig.update(data=(data_mapbox, fig.data[1], data_cursor))
                fig.layout.sliders[0].update(active=s)
                # Converts the figure to an image
                gif_frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png",
                                                                         width=1600,
                                                                         height=900))))
            # Assembles and saves the GIF
            gif_frames[0].save(
                save_gif_to,
                save_all=True,
                append_images=gif_frames[1:],
                optimize=True,
                duration=1250,
                loop=0
            )
        if show_fig:
            fig.show()
