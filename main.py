from participant_gaze_data_manager import ParticipantGazeDataManager
from training_npy_generator import GazeDataProcessor
import os
import glob

def convert_recordings_to_npy(data_path: str, task: str) -> None:
    """
    Convert recordings from task to npy files.

    This function loads the recordings from the SDMT task and converts them to npy files.
    The recordings are loaded using the ParticipantGazeDataManager class and then processed
    using the GazeDataProcessor class.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the recordings.
    task : str
        The name of the task (in this case, SDMT).
    """
    data_managers = []
    for group in ["pwMS", "HC"]:
        for subject_name in glob.glob(os.path.join(data_path, group, "*")):
            if not os.path.isdir(subject_name):
                continue
            if task not in os.listdir(subject_name):
                continue
            data_managers.append(
                ParticipantGazeDataManager(subject_name, data_path, task, group, clean_gaze_data=True)
            )
    processor = GazeDataProcessor(data_managers)
    processor.process_gaze_data("data_for_training", show=True)

data_path = "/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior"
convert_recordings_to_npy(data_path, "SDMT")