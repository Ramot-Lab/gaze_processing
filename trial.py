from FixationHandler import Fixation
from PanelSymbols import Symbol
from SearchFinder import Search


class Trial:
    def __init__(self, idx, triggering_symbol: Symbol, 
                 relevant_fixations: list[Fixation], searches: list[Search], start_time, end_time):
        self.idx = idx
        self.triggering_symbol = triggering_symbol
        self.relevant_fixations = relevant_fixations
        self.searches = searches
        self.start_time = start_time
        self.end_time = end_time

    @property
    def duration(self):
        return self.end_time - self.start_time