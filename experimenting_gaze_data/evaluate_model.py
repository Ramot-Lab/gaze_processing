from experimenting_gaze_data.auto_encoder_DTC import Autoencoder
import torch
import os
import numpy as np
import json
from data_processing.data_loader import load_npy_files
from data_processing.data_loader import GazeDataLoader, FixationDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

class ModelEvaluator:

    def __init__(self, config_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.model = Autoencoder(latent_dim=self.config['latent_dim']).to(self.device)
        self.model.load_state_dict(torch.load(self.config['model_path']))
        self.model.eval()

    def run_evaluation(self, show=True):
        self.config['split_seqs']=True
        self.config['augment']=False
        self.config['batch_size']=1
        data = load_npy_files(self.config['val_data_path'], self.config, "val")
        batch_size = self.config['batch_size']
        dataset = FixationDataset(config = self.config, gaze_data = data)
        loader = GazeDataLoader(dataset, batch_size=batch_size,
                                    shuffle=True)
        results = []
        for step, data in enumerate(loader):
            ##Prepare data
            inputs, targets, input_percentages, target_sizes, _ = data
            inputs = Variable(inputs)
            y_ = Variable(targets)
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            ##Forward Pass
            latent_representation, y = self.model(inputs)
            results.append((latent_representation.cpu().detach().numpy(), y.cpu().detach().numpy(), inputs.cpu().detach().numpy()))
        if show:
            self.plot_reconstruction_results(results)
            self.plot_clustering_results(results)
        return results
    
    def plot_reconstruction_results(self, results): 
        # show 4 examples of reconstruction results.
        # every plot shows the original gaze in blue and reconstructed in red lines, horizontal in one plot and vertical in another plot
        # every data is <t, horizontal, vertical, ...>
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for i in range(2):
            latent_representation, reconstructed, original = results[i]
            latent_representation, reconstructed, original =latent_representation, reconstructed[0][0], original[0][0]
            axs[i, 0].plot( original[0], marker='o', color='blue', label='original')
            axs[i, 0].plot(reconstructed[0], marker='o', color='red', label='reconstructed')
            axs[i, 0].legend()
            axs[i, 0].set_xlabel('time')
            axs[i, 0].set_ylabel('horizontal')
            axs[i, 0].grid()
            axs[i, 1].plot( original[1], marker='o', color='blue', label='original')
            axs[i, 1].plot(reconstructed[1], marker='o', color='red', label='reconstructed')
            axs[i, 1].legend()
            axs[i, 1].set_xlabel('time')
            axs[i, 1].set_ylabel('vertical')
            axs[i, 1].grid()
        plt.show()  

    def plot_clustering_results(self, results):
        # generates a UMAP out of the set of latent representations
        # and plots the results
        latent = [l for l, _, _ in results]
        latent = np.concatenate(latent)
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(latent)

        # Plot the results in 3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Optionally, color the points (e.g., based on clusters or labels)
        # Here, we use random labels for demonstration
        labels = np.random.randint(0, 3, size=latent.shape[0])  # 3 random classes
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                            c=labels, cmap='Spectral', s=50, alpha=0.8)

        # Add a color bar for interpretation
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Labels')

        # Add axis labels and title
        ax.set_xlabel('PCA Dimension 1')
        ax.set_ylabel('PCA Dimension 2')
        ax.set_zlabel('PCA Dimension 3')
        plt.title('PCA Projection into 3D Space')

        # Show the plot
        plt.show()