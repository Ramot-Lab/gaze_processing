from participant_gaze_data_manager import ParticipantGazeDataManager
from data_processing.training_npy_generator import MultipleGazeDataProcessor
import os
import glob
import numpy as np
from experimenting_gaze_data.trainer import Trainer

def convert_recordings_to_npy_SDMT(data_path: str, task: str, output_path: str) -> None:
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
            try:
                par_data = ParticipantGazeDataManager(subject_name, data_path, task, group,  clean_gaze_data=True)
            except:
                print(f"an error with processing subject : {subject_name}")
                continue
            panel_output_path = os.path.join(output_path, name)
            os.makedirs(panel_output_path, exist_ok=True)
            for panel in par_data.matched_data.keys():
                if os.path.exists(os.path.join(panel_output_path, f"{panel}_{task}_gaze_annotated_data.npy")):
                    continue # "Lazy" annotation
                annotated_data = par_data.annotate_gaze_events("model_based", panel)
                np.save(os.path.join(panel_output_path, f"{panel}_{task}_gaze_annotated_data.npy"), annotated_data)
                print(f"Saved {panel} panel data for {name}")


def convert_recordings_to_npy_KD(data_path: str, task: str, output_path: str) -> None:
    """
       Convert recordings from the KD task to npy files.

       Parameters
       ----------
       data_path : str
           The path to the directory containing the recordings.
       task : str
           The name of the task (e.g., "KD").
       output_path : str
           Path to save processed output files.
       """
    os.makedirs(output_path, exist_ok=True)
    for group in ["pwMS", "HC"]:
        group_path = os.path.join(data_path, group)
        for subject_path in glob.glob(os.path.join(group_path, "*")):
            subject_name = os.path.basename(subject_path)
            if not os.path.isdir(subject_path) or task not in os.listdir(subject_path):
                continue

            try:
                print(f"Processing subject: {subject_name} (Group: {group})")

                # Initialize ParticipantGazeDataManager
                par_data = ParticipantGazeDataManager(subject_name, data_path, task, group, clean_gaze_data=True)

                # Set output path for this participant
                participant_output_path = os.path.join(output_path, subject_name)
                os.makedirs(participant_output_path, exist_ok=True)

                # Process each panel in the matched data
                for panel in par_data.matched_data.keys():
                    output_file = os.path.join(participant_output_path, f"{panel}_{task}_gaze_annotated_data.npy")
                    if os.path.exists(output_file):
                        continue  # Skip already processed panels

                    # Annotate and save gaze data
                    annotated_data = par_data.annotate_gaze_events("model_based", panel)
                    np.save(output_file, annotated_data)
                    print(f"Saved data for panel {panel} (Participant: {subject_name})")

            except Exception as e:
                print(f"Error processing subject {subject_name} (Group: {group}): {e}")

def convert_recordings_to_npy(data_path: str, task: str, output_path: str) -> None:
    """
    Convert recordings from a task to npy files.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the recordings.
    task : str
        The name of the task (e.g., "KD", "SDMT").
    output_path : str
        Path to save processed output files.
    """
    os.makedirs(output_path, exist_ok=True)
    for group in ["pwMS", "HC"]:
        group_path = os.path.join(data_path, group)
        for subject_path in glob.glob(os.path.join(group_path, "*")):
            subject_name = os.path.basename(subject_path)
            if not os.path.isdir(subject_path) or task not in os.listdir(subject_path):
                continue

            try:
                print(f"Processing subject: {subject_name} (Group: {group})")

                # Initialize ParticipantGazeDataManager
                par_data = ParticipantGazeDataManager(subject_name, data_path, task, group, clean_gaze_data=True)

                # Set output path for this participant
                participant_output_path = os.path.join(output_path, group, subject_name)
                os.makedirs(participant_output_path, exist_ok=True)

                # Process each panel in the matched data
                for panel in par_data.matched_data.keys():
                    output_file = os.path.join(participant_output_path, f"{panel}_{task}_gaze_annotated_data.npy")
                    if os.path.exists(output_file):
                        continue  # Skip already processed panels

                    # Annotate and save gaze data
                    annotated_data = par_data.annotate_gaze_events("model_based", panel)
                    np.save(output_file, annotated_data)
                    print(f"Saved data for panel {panel} (Participant: {subject_name})")

            except Exception as e:
                print(f"Error processing subject {subject_name} (Group: {group}): {e}")


if __name__ == "__main__":
    import sys
    task_to_run = "generate_data"
    if task_to_run == "generate_data":
        data_path = "/Volumes/WIS_Ido/Results/Behavior/By_Tasks/KD"
        output_path = "/Volumes/WIS_Ido/Results/Analysis/King_Devick/Prior_analysis/2412/pwMS"

        convert_recordings_to_npy(data_path, "KD", output_path)
        #convert_recordings_to_npy_SDMT(data_path, "SDMT", output_path)
    elif task_to_run == "train":
        trainer = Trainer()
        trainer.run()
    else:
        print("Unknown command, choose from ['generate_data'- for data generation, 'train' - for auto incoder model training]")

