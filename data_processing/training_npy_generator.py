import os
import numpy as np
from constants import deg_per_pixel, KEY_TOBII_DATA
import matplotlib.pyplot as plt
import copy

class MultipleGazeDataProcessor:
    def __init__(self, participant_gaze_managers):
        """
        Initialize the processor with a list of ParticipantGazeDataManager objects.

        Args:
            participant_gaze_managers (list): List of ParticipantGazeDataManager instances.
        """
        self.participant_gaze_managers = participant_gaze_managers

    
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

                    recording_data_preprocessor = Recoring2GazeNetProcessor(tobii_data)
                    structured_data = recording_data_preprocessor.get_processed_data()                  
                    if show:
                        self.plot_manipulated_data(structured_data)
                    # Save as .npy file
                    output_file = os.path.join(participant_output_path, f"{panel_key}_gaze_data.npy")
                    np.save(output_file, structured_data)
                    print(f"Processed gaze data saved for {manager.name}, {panel_key} at {output_file}")


class Recoring2GazeNetProcessor:
    """
    converts single tobii recording np array to gazenet format
    """
    def __init__(self, tobii_data):
        self.tobii_data = tobii_data

    def preprocess_tobii_to_degrees(self, tobii_data):
        """
        Converts Tobii data to degrees. Assumes Tobii data contains gaze points as (x, y) coordinates
        ranging from 0 to 1.

        Args:
            tobii_data (numpy.ndarray): n x 3 array of Tobii data.
 
        Returns:
            numpy.ndarray: Converted gaze data in degrees.
        """
        screen_width, screen_height = 1920, 1080  #Ramot Lab's screen dimensions
        tobii_degrees_data = copy.deepcopy(tobii_data)
        tobii_degrees_data[:, 0] = (tobii_data[:, 0] * screen_width) - screen_width//2
        tobii_degrees_data[:, 1] = (tobii_data[:, 1] * screen_height) - screen_height//2
        tobii_degrees_data[:, 0] *= deg_per_pixel
        tobii_degrees_data[:, 1] *= deg_per_pixel
        return np.concatenate([tobii_degrees_data,tobii_data[:, :2]], axis=1) #add valid data
    
    def align_data_structure(self, tobii_data):
        """
        Aligns and structures Tobii gaze data for further processing.

        Converts raw Tobii data into a structured numpy array with specified fields.
        The structured array contains columns for time, x and y coordinates, event 
        labels, and status. The time is adjusted to start from zero, and status flags 
        are set to indicate the validity of data points.

        Args:
            tobii_data (numpy.ndarray): An array of Tobii gaze data with columns 
                                        representing x and y coordinates, and time.

        Returns:
            numpy.ndarray: A structured array with fields ('t', 'x', 'y', 'evt', 'status'),
                        where 't' is time, 'x' and 'y' are coordinates, 'evt' is event
                        label (default 0), and 'status' is data validity (1 for valid, 
                        0 for invalid points).
        """
        structured_data = np.empty(len(tobii_data), dtype=[
                        ('t', 'f8'),   # Time
                        ('x', 'f8'),   # X-coordinate
                        ('y', 'f8'),   # Y-coordinate
                        ('evt', 'i4'), # Event labels (default 0 for now)
                        ('status', 'i4'), # Status (validity, default 1 for now)
                        ('x_original', 'f8'),   # X-coordinate-original
                        ('y_original', 'f8'),   # Y-coordinate-original
                    ])
        
        # Populate structured data
        structured_data['t'] = tobii_data[:, 2] - tobii_data[0,2]  # Time column
        structured_data['x'] = tobii_data[:, 0]  # X-coordinate
        structured_data['y'] = tobii_data[:, 1]  # Y-coordinate
        structured_data['evt'] = 0  # Default event label
        structured_data['status'] = 1  # Mark all as valid initially
        structured_data["status"][np.isnan(structured_data["x"])] = 0 
        structured_data["x_original"] = tobii_data[:, 3]
        structured_data["y_original"] = tobii_data[:, 4]
        return structured_data

    def get_processed_data(self):
        """
        Returns a structured numpy array containing the processed Tobii data in degrees.

        The structured array contains fields for time, x and y coordinates, event labels,
        and status. The time is adjusted to start from zero, and status flags are set to
        indicate the validity of data points.

        Returns:
            numpy.ndarray: A structured array with fields ('t', 'x', 'y', 'evt', 'status'),
                        where 't' is time, 'x' and 'y' are coordinates, 'evt' is event
                        label (default 0 for now), and 'status' is data validity (1 for valid,
                        0 for invalid points).
        """
        degreezed_data = self.preprocess_tobii_to_degrees(self.tobii_data)
        structured_data = self.align_data_structure(degreezed_data)
        return structured_data
    