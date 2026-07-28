import re

import numpy as np

from constants import KEY_PANEL_MESSAGES, KEY_REACTION_TIMES, SECONDS_TO_MICROSECOND_FACTOR
from participant_gaze_data_manager import ParticipantGazeDataManager

class Press:
    def __init__(self, idx, time):
        self.idx = idx # press 0 is the start of panel, last press is end of panel
        self.time = time


class MessageInfo:
    def __init__(self, panel_name, start_time, end_time, presses):
        self.panel = panel_name
        self.start_time = start_time
        self.end_time = end_time
        self.presses = presses  # list of press objects

    def __repr__(self):
        return f"<Messages of panel: {self.panel}>"
    

class PanelMessages:
    def __init__(self, panel:str, subject_data: ParticipantGazeDataManager):
        """
        Initialize a list of panel message objects from a global messages array.

        Parameters
        ----------
        all_messages : np.ndarray
            2D array where each row = [timestamp, message_text]
        """
        self.subject_data = subject_data
        self.message_info :MessageInfo = None  # list messageInfo objects
        messages = subject_data.matched_data[panel][KEY_PANEL_MESSAGES]
        if messages is None or messages.size < 4:
            raise ValueError(f"No messages found for subject {subject_data.name} for panel '{panel}'")
        else:
            start_time = messages[0, 0]
            end_time = messages[-1, 0]

            presses = []
            for ts, text in messages:
                if isinstance(text, str) and text.startswith("press "):
                    match = re.search(r"press (\d+)", text)
                    if match:
                        presses.append((int(match.group(1)), ts - start_time))

            # Older recordings never logged "press N" messages at all for some panels.
            # Fall back to task_data.reaction_times_img_test_{PANEL}: one value per press,
            # each the number of seconds since the *previous* press (first value is since
            # the panel started, i.e. the 'panel number N' message). Reconstructing
            # absolute press times from their cumulative sum lets these recordings be
            # included instead of dropped.
            if not presses:
                reaction_times = subject_data.matched_data[panel].get(KEY_REACTION_TIMES)
                if reaction_times is not None and len(reaction_times) > 0:
                    cumulative_seconds = np.cumsum(reaction_times)
                    presses = [(i + 1, t * SECONDS_TO_MICROSECOND_FACTOR) for i, t in enumerate(cumulative_seconds)]

            # Convert to Press objects
            press_objs = [Press(idx=p[0], time=p[1]) for p in presses]

            # --- Add synthetic first and last press ---
            # First press: start of panel (time = 0)
            press_objs.insert(0, Press(idx=0, time=0))
            # Last press: end of panel (time = end_time - start_time)
            press_objs.append(Press(idx=len(press_objs), time=end_time - start_time))

            # Create and store MessageInfo
            self.message_info = MessageInfo(
                panel_name=panel,
                start_time=0,
                end_time=end_time - start_time,
                presses=press_objs
            )

    def __repr__(self):
        return f"<PanelMessages: {len(self.message_info)} panels>"
    
