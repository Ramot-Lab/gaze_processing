from datetime import datetime

matching_dictionary_keys = ['a3', '0', 'l3', 'l4', 'i1', 'a5']
matching_dictionary = {'a3': 
                       {'tobii_data': 
                        [[4.91818964e-01, 4.87222016e-01, 1.31334979e+12], [4.91764009e-01, 4.88000001e-01, 1.31344024e+12]],
                        'task_panel_img': 
                        '/Volumes/ramot/rotation_students/Noam_M/Results/Behavior/panels_images/SDMT/combined_testable_a3.jpg', 
                        'audio_data': '/Volumes/ramot/rotation_students/Noam_M/Results/Behavior/pwMS/AG562/SDMT/img_test_A3_strikes_74.wav', 
                        'strike_score': 74, 
                        'recording_date': datetime.datetime(2025, 1, 21, 14, 13, 16)}, 

                        '0': {'tobii_data': [4.84975249e-01, 4.93774414e-01, 1.31345221e+12]}
}


# matching_dictionary = subject.matched_data
#     task_data = matching_dictionary[task_code]
#     eye_data = task_data[KEY_TOBII_DATA]
#     img = cv2.imread(task_data[KEY_TASK_PANEL_IMG])