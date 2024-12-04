from participant_gaze_data_manager import ParticipantGazeDataManager
from data_processing.training_npy_generator import MultipleGazeDataProcessor
import os
import glob
import numpy as np
from experimenting_gaze_data.trainer import Trainer

def convert_recordings_to_npy(data_path: str, task: str, output_path: str) -> None:
    """
    Convert recordings from task to npy files.

    This function loads the recordings from the SDMT task and converts them to npy files.
    The recordings are loaded using the ParticipantGazeDataManager class and then processed
    using the MultipleGazeDataProcessor class.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the recordings.
    task : str
        The name of the task (in this case, SDMT).
    """
    data_managers = []
    os.makedirs(output_path, exist_ok=True)
    for group in ["pwMS", "HC"]:
        for subject_name in glob.glob(os.path.join(data_path, group, "*")):
            name = os.path.split(subject_name)[-1]
            if not os.path.isdir(subject_name):
                continue
            if task not in os.listdir(subject_name):
                continue
            par_data = ParticipantGazeDataManager(subject_name, data_path, task, group, clean_gaze_data=True)
            panel_output_path = os.path.join(output_path, name)
            os.makedirs(panel_output_path, exist_ok=True)
            for panel in par_data.matched_data.keys():
                annotated_data = par_data.annotate_gaze_events("model_based", panel)
                np.save(os.path.join(panel_output_path, f"{panel}_{task}_gaze_annotated_data.npy"), annotated_data)
    #         data_managers.append(
    #             ParticipantGazeDataManager(subject_name, data_path, task, group, clean_gaze_data=True)
    #         )
    # processor = MultipleGazeDataProcessor(data_managers)
    # processor.process_gaze_data("data_for_training", show=True)

if __name__ == "__main__":
    data_path = "/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior"
    output_path = "data_for_training"
    convert_recordings_to_npy(data_path, "SDMT", output_path)
    # trainer = Trainer()
    # trainer.run()

# data_path = "/Volumes/labs/ramot/rotation_students/Nitzan_K/MS/Results/Behavior"
# convert_recordings_to_npy(data_path, "SDMT")
# convert_recordings_to_npy(data_path, "KD") 