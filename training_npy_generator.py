import os
import numpy as np
from constants import deg_per_pixel, KEY_TOBII_DATA
import matplotlib.pyplot as plt

class GazeDataProcessor:
    def __init__(self, participant_gaze_managers):
        """
        Initialize the processor with a list of ParticipantGazeDataManager objects.

        Args:
            participant_gaze_managers (list): List of ParticipantGazeDataManager instances.
        """
        self.participant_gaze_managers = participant_gaze_managers

    def preprocess_tobii_to_degrees(self, tobii_data):
        """
        Converts Tobii data to degrees. Assumes Tobii data contains gaze points as (x, y).

        Args:
            tobii_data (numpy.ndarray): n x 3 array of Tobii data.

        Returns:
            numpy.ndarray: Converted gaze data in degrees.
        """
        screen_width, screen_height = 1920, 1080  # Example screen resolution
        # deg_per_pixel = 0.03  # Example scaling factor
        tobii_data[:, 0] = (tobii_data[:, 0] * screen_width) - screen_width//2
        tobii_data[:, 1] = (tobii_data[:, 1] * screen_height) - screen_height//2
        tobii_data[:, 0] *= deg_per_pixel
        tobii_data[:, 1] *= deg_per_pixel
        return tobii_data
    
    def plot_manipulated_data(self, tobii_data):
        fig = plt.figure(figsize=(10,6))
        ax00 = plt.subplot2grid((2, 2), (0, 0))
        ax10 = plt.subplot2grid((2, 2), (1, 0), sharex=ax00)
        ax01 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        ax00.plot(tobii_data['t'], tobii_data['x'], '-')
        ax10.plot(tobii_data['t'], tobii_data['y'], '-')
        ax01.plot(tobii_data['x'], tobii_data['y'], '-')
        

    def process_gaze_data(self, output_path, show = False):
        """
        Processes the gaze data from all participants and saves as .npy files.

        Args:
            output_path (str): Directory where the .npy files will be saved.
        """
        os.makedirs(output_path, exist_ok=True)

        for manager in self.participant_gaze_managers:
            participant_output_path = os.path.join(output_path, manager.name)
            os.makedirs(participant_output_path, exist_ok=True)

            for panel_key in manager.matched_data.keys():
                if KEY_TOBII_DATA in manager.matched_data[panel_key]:
                    tobii_data = manager.matched_data[panel_key][KEY_TOBII_DATA]

                    # Preprocess gaze data to degrees
                    processed_gaze_data = self.preprocess_tobii_to_degrees(tobii_data)

                    # Create structured array with required fields
                    structured_data = np.empty(len(processed_gaze_data), dtype=[
                        ('t', 'f8'),   # Time
                        ('x', 'f8'),   # X-coordinate
                        ('y', 'f8'),   # Y-coordinate
                        ('evt', 'i4'), # Event labels (default 0 for now)
                        ('status', 'i4') # Status (validity, default 1 for now)
                    ])
                    
                    # Populate structured data
                    structured_data['t'] = processed_gaze_data[:, 2] - processed_gaze_data[0,2]  # Time column
                    structured_data['x'] = processed_gaze_data[:, 0]  # X-coordinate
                    structured_data['y'] = processed_gaze_data[:, 1]  # Y-coordinate
                    structured_data['evt'] = 0  # Default event label
                    structured_data['status'] = 1  # Mark all as valid initially
                    structured_data["status"][np.isnan(structured_data["x"])] = 0 
                    if show:
                        self.plot_manipulated_data(structured_data)
                    # Save as .npy file
                    output_file = os.path.join(participant_output_path, f"{panel_key}_gaze_data.npy")
                    np.save(output_file, structured_data)
                    print(f"Processed gaze data saved for {manager.name}, {panel_key} at {output_file}")
