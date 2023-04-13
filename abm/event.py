"""
Clément Dauvilliers - April 13th 2023
Implements the EventQueue class.
"""
from enum import Enum
from collections import defaultdict


class EventType(Enum):
    """
    Defines the possible types of events.
    """
    MOBILITY_REDUCTION = 1
    MOBILITY_RESET = 2


class EventQueue:
    """
    The EventQueue class is a collection of simulation events which will happen at a
    certain date ().
    An event can be anything that triggers a reaction within the model;
    for example, reducing the activity of an agent or restoring it are events.

    An event is defined as a pair (type, info):
    - type is a value of the EventType enumeration;
    - info is a list/tuple containing all information that the model requires to manage the event.
        Its structure and content is specific to each type of event.
    """
    def __init__(self):
        """
        Creates an empty EventQueue.
        """
        # The "Queue" is not actually a basic queue: it needs to be ordered by date.
        # Therefore, we'll use a dictionary {(day, period): list of events}.
        self.events = defaultdict(list)

    def add_event(self, day, period, event_type, event_info):
        """
        Adds a new event to the queue.
        Parameters
        ----------
        day: simulation day on which the event should happen.
        period: period of the specified day on which the event should happen.
        event_type: value of the EventType enum.
        event_info: tuple or list containing all information regarding the event that the model
            needs to manage the event.
        """
        self.events[(day, period)].append((event_type, event_info))

    def poll_events(self, day, period):
        """
        Returns the list of all events that should happen on a given date.
        Warning: this removes the events that were returned from the queue.
        Parameters
        ----------
        day: simulation day for which the events should be retrieved.
        period: period of the specified day.

        Returns
        -------
        A list of events.
        An event is defined as a pair (type, info):
        - type is a value of the EventType enumeration;
        - info is a list/tuple containing all information that the model requires to manage the event.
            Its structure and content is specific to each type of event.
        """
        if (day, period) not in self.events:
            return []
        return self.events.pop((day, period))