
import argparse
import json
from data_processing.data_loader import load_npy_files, GazeDataLoader, FixationDataset
import copy
from experimenting_gaze_data.auto_encoder_DTC import Autoencoder
import torch
import torch.nn as nn
from torch.autograd import Variable 
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

    
class Trainer:
    def __init__(self):
        self.args = self.get_arguments()
        config_path = self.args.config_file
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        os.makedirs(self.config["logs_path"], exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model = Autoencoder(latent_dim = self.config['latent_dim']).to(self.device)
        self.reconstruction_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.loader_train = self.load_data(self.config["train_data_path"], self.config, "train")
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config['learning_rate'], 
        #                                                      steps_per_epoch=len(self.loader_train), epochs=self.args.num_epochs)
        self.config_val = copy.deepcopy(self.config)
        self.config_val['split_seqs']=False
        self.config_val['batch_size']=1
        self.config_val['augment']=False
        self.loader_val = self.load_data(self.config["val_data_path"], self.config_val, "val")
        self.best_score = np.inf
        self.best_model = None
        self.train_loss = []
        self.val_loss = []

    def get_arguments(self):
        parser = argparse.ArgumentParser(description='gazeNet: End-to-end eye-movement event detection with deep neural networks')
        parser.add_argument('--config_file', type=str,
                            help='training self.configuration file')
        parser.add_argument('--num_epochs', default=500, type=int,
                            help='Number of epochs to train')
        parser.add_argument('--seed', type=int, default=220617, help='seed')
        return parser.parse_args()

    def load_data(self, data_path, config, train_val):
        data = load_npy_files(data_path, config, train_val)
        batch_size = self.config['batch_size']
        dataset = FixationDataset(config = self.config, gaze_data = data)
        loader = GazeDataLoader(dataset, batch_size=batch_size,
                                    shuffle=True)
        print(f"{train_val} dataset loaded with {dataset.__len__()} samples")
        return loader

    def _compute_clustering_loss(self, latent_representation, n_clusters):
        batch_size, latent_dim = latent_representation.size()
        device = latent_representation.device
        LARGE_NUMBER = 10000
        # Step 1: Initialize centroids randomly from the data points
        indices = torch.randint(0, batch_size, (n_clusters,), device=device)
        centroids = latent_representation[indices]  # [n_clusters, latent_dim]

        for _ in range(self.config['kmeans_iterations']):  # Number of iterations to refine centroids
            # Step 2: Compute distances to centroids and assign clusters
            distances = torch.cdist(latent_representation, centroids)  # [batch_size, n_clusters]
            assignments = torch.argmin(distances, dim=1)  # Assign each point to the closest centroid

            # Step 3: Update centroids
            # Group points by cluster and compute their mean
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=device)

            for k in range(n_clusters):
                mask = (assignments == k)
                if mask.any():
                    new_centroids[k] = latent_representation[mask].mean(dim=0)
                    counts[k] = mask.sum()

            # Handle empty clusters by reinitializing
            empty_clusters = (counts == 0)
            if empty_clusters.any():
                empty_indices = torch.where(empty_clusters)[0]
                new_centroids[empty_indices] = latent_representation[torch.randint(0, batch_size, (empty_indices.size(0),), device=device)]

            centroids = new_centroids

        # Compute clustering loss (mean squared distance to assigned centroids)
        assigned_centroids = centroids[assignments]  # [batch_size, latent_dim]
        clustering_loss = F.mse_loss(latent_representation, assigned_centroids)

        # Compute centroid separation loss
        pairwise_distances = torch.cdist(centroids, centroids)  # [n_clusters, n_clusters]
        mask = torch.eye(n_clusters, device=centroids.device, dtype=torch.int)  # Diagonal mask
        # pairwise_distances_with_mask = pairwise_distances + (mask * LARGE_NUMBER)   # Mask diagonal elements to avoid self-distance
        repulsion_loss_ = 1 / (pairwise_distances + 1e-6)  # Add small value to avoid division by zero
        repulsion_loss = torch.sum(torch.triu(repulsion_loss_, diagonal=1))  # Sum only unique pairs

        return 10000*clustering_loss + 0.001*repulsion_loss


    def run(self):
        device = self.device
        # loading data
        print("loading data...")   
        #training loop
        for epoch in range(0, self.args.num_epochs+1): 
            running_loss = 0
            iterator = tqdm(self.loader_train)
            for step, data in enumerate(iterator):
                global_step = len(self.loader_train)*(epoch-1) + step
                ##Prepare data
                inputs, targets, input_percentages, target_sizes, _ = data
                inputs = Variable(inputs)
                y_ = Variable(targets)
                if device != "cpu":
                    inputs = inputs.to(device)
                    y_ = y_.to(device)
                ##Forward Pass
                latent_representation, y = self.model(inputs)
                # recornstuction_loss
                reconstruction_loss = self.reconstruction_criterion(y,inputs)
                # clustering loss
                if self.config["use_clustering_loss"]:
                    clustering_loss = self._compute_clustering_loss(latent_representation, self.config['n_clusters'])
                    loss =  (100*reconstruction_loss) + clustering_loss
                else:
                    loss = 100 * reconstruction_loss
                ##Backward pass
                if np.isfinite(loss.item()):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # self.scheduler.step()
                    running_loss += loss.item()
            print(f"Epoch: {epoch}, Loss: {running_loss/len(self.loader_train)}")
            if epoch > self.config['test_interval'] and epoch % self.config['test_interval'] == 0:
                self.train_loss.append(loss.item())
                self.run_validation(self.loader_val, epoch)
        self.run_validation(self.loader_val, epoch)

    def plot_loss(self, model_path, epoch):
        plt.clf()
        plt.plot(self.train_loss, label='train loss')
        plt.plot(self.val_loss, label='val loss')
        plt.xlabel('Epoch')
        plt.ylim(0, 50)
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(model_path, f'loss_{epoch}.png'))

    def run_validation(self, loader_val, epoch):
        total_loss = 0
        self.model.eval()
        for step, data in enumerate(loader_val):
            ##Prepare data
            inputs, targets, input_percentages, target_sizes, _ = data
            inputs = Variable(inputs)
            y_ = Variable(targets)
            if self.device != "cpu":
                inputs = inputs.to(self.device)
                y_ = y_.to(self.device)
            ##Forward Pass
            latent_representation, y = self.model(inputs)
            reconstruction_loss = self.reconstruction_criterion(y,inputs)
            # clustering loss
            if self.config["use_clustering_loss"]:
                clustering_loss = self._compute_clustering_loss(latent_representation, self.config['n_clusters'])
                loss = (100*reconstruction_loss) + clustering_loss
            else:
                loss = 100 * reconstruction_loss
            total_loss += loss.item()
        if (total_loss/len(loader_val)) < self.best_score:
            self.best_score = loss
            self.best_model = copy.deepcopy(self.model)
            print("best model updated, best score: %f" % self.best_score)
            model_output_path = self.save_model_checkpoint() 
            self.plot_loss(model_output_path, epoch)
        print("Validation loss: %f" % loss.item())
        self.val_loss.append(loss.item())
        self.model.train()
        return           

    def save_model_checkpoint(self):
        folder_name = datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join(self.config["logs_path"], folder_name)
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.best_model.state_dict(), os.path.join(folder_path, f"model_{self.best_score}.pth"))

        # save the training config file for easier reproducibility
        if not os.path.exists(os.path.join(folder_path, "config.json")):
            with open(os.path.join(folder_path, "config.json"), "w") as f:  
                json.dump(self.config, f, indent=4)
        return folder_path

