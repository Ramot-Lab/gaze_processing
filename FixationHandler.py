import pandas as pd
from constants import FIXATION_IDX, SACCADE_IDX
from RoiFinder import *


class Fixation:
    def __init__(self, start_time, end_time, position, microsaccades = []):
        self.start_time = start_time
        self.end_time = end_time
        self.position = position  # (x, y) coordinates
        self.microsaccades = microsaccades


    def duration(self):
        return self.end_time - self.start_time


class FixationHandler:
    def __init__(self, data):
        self.fixations : list[Fixation] = []
        self.process_fixations(data)


    def get_fixations(self):
        return self.fixations

    
    #--------------------------------------------
    #             Fixation making
    #--------------------------------------------
       
    def _add_fixation(self, fixation_events: list[tuple]):
        """
        Convert a list of fixation events into a Fixation object.
        """
        df = pd.DataFrame(fixation_events, columns=['t', 'x', 'y'])
        start_time = df['t'].iloc[0]
        end_time = df['t'].iloc[-1]
        mean_position = (df['x'].mean(), df['y'].mean())
        fixation = Fixation(start_time=start_time,
                            end_time=end_time,
                            position=mean_position,
                            microsaccades=df)
        self.fixations.append(fixation)
    
    def process_fixations(self, data):
        """
        Convert raw gaze data into a list of Fixation objects.
        
        Parameters
        ----------
        data : iterable of rows (t, x, y, valid, event_type)
            Raw gaze events.
        """
        current_fixation = []
        
        for _, row in data.iterrows():
            t, x, y, valid, event_type = row
            if not valid:
                continue
            if event_type == FIXATION_IDX:
                # accumulate fixation events
                current_fixation.append((t, x, y))
            elif event_type == SACCADE_IDX and current_fixation:
                # end of a fixation block, create Fixation object
                self._add_fixation(current_fixation)
                current_fixation = []

        # in case the last block is a fixation and no SACCADE follows
        if current_fixation:
            self._add_fixation(current_fixation)

        return self.fixations



        
class FixationSaccadeVisualization:
    def __init__(self, fixations: list[Fixation]):
        self.fixations = fixations