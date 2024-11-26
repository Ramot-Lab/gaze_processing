import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_correlation_scatter(array, l_size, subject_amount):
    """
    Plots scatter plot of the correlation matrix
    array: np.array, matrix of correlations, shape n_repetitions,
    n_trials_list: list/array, of n_trials used to compute the correlations from
    ax: matplotlib axis
    """
    plt.hist(array, bins=100, label=f"L size: {l_size}")
    plt.title(f"Reliability hist over {len(array)} samples")
    plt.show()


def calculate_reliability(data, L_size, repetitions, min_value, plot = True):
    """
    data: a 2d matrix of N subjects as rows and M samples as columns
    L_size : the size of the vector to random from
    repetitions: number of times the repetitions.
    """
    array_corr = np.zeros(repetitions)
    subjects_amount = len(data)
    for i in range(repetitions):
        
        shuffle_data = np.random.permutation(np.arange(min_value))
        selected_matrix_1 = []
        selected_matrix_2 = []
        for participant_data in range(len(data)):
            matrix_1 = np.array(data[participant_data])[shuffle_data[:L_size]]
            matrix_2 = np.array(data[participant_data])[shuffle_data[L_size:2*L_size]]
            selected_matrix_1.append( np.mean(matrix_1))
            selected_matrix_2.append(np.mean(matrix_2))

        array_corr[i] = np.corrcoef(selected_matrix_1, selected_matrix_2)[0,1]
    # fig, ax = plt.subplots(1, 1, figsize=(8,6))
    if plot:
        plot_correlation_scatter(array_corr, L_size, subjects_amount)
    return np.mean(array_corr), L_size

def calculate_reliability_distribution(data, min_l, max_l, repetitions, min_value):
    results = []
    for l_value in range(max(5,min_l), max_l):
        mean_value, _ = calculate_reliability(data, l_value, repetitions, min_value, plot = True)
        results.append(mean_value)
    return results

if __name__=="__main__":
    random_data = np.random.rand(10,1000)
    calculate_reliability(random_data, 20, 1000, 30)


        