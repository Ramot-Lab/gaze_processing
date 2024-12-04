
import argparse
import json
from data_processing.data_loader import load_npy_files, EMDataset, GazeDataLoader
import copy
from experienting_gaze_data.auto_encoder import AutoEncoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np
import os


class RowwiseMSELoss(nn.Module):
    def __init__(self):
        super(RowwiseMSELoss, self).__init__()

    def forward(self, input, output):
        # Ensure input and output have the same shape (batch_size, num_rows, 2)
        assert input.shape == output.shape, "Input and output shapes must match"
        
        # Compute the squared differences row-wise
        diff = input - output
        squared_diff = diff.pow(2).sum(dim=2)  # Sum squared differences for x and y
        mse = squared_diff.mean()  # Average over rows and batch
        return mse
    
class Trainer:
    def __init__(self):
        self.args = self.get_arguments()
        config_path = self.args.config_file
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(self.config, latent_dim = self.config['latent_dim'])
        self.criterion = RowwiseMSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.loader_train = self.load_data(self.config["train_data_path"], self.config, "train")
        self.config_val = copy.deepcopy(self.config)
        self.config_val['split_seqs']=False
        self.config_val['batch_size']=1
        self.config_val['augment']=False
        self.val_dataset = self.load_data(self.config["val_data_path"], self.config_val, "val")
        self.best_score = np.inf
        self.best_model = None

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
        dataset_train = EMDataset(config = self.config, gaze_data = data)
        loader_train = GazeDataLoader(dataset_train, batch_size=batch_size,
                                    shuffle=True)
        return loader_train

    def run(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # loading data
        print("loading data...")   
        #training loop
        for epoch in range(0, self.args.num_epochs+1): 
            iterator = tqdm(self.loader_train)
            for step, data in enumerate(iterator):
                global_step = len(self.loader_train)*(epoch-1) + step
                ##Prepare data
                inputs, targets, input_percentages, target_sizes, _ = data
                inputs = Variable(inputs)
                y_ = Variable(targets)
                if device == "cuda":
                    inputs = inputs.cuda()
                    y_ = y_.cuda()

                ##Forward Pass
                y = self.model(inputs)
                loss = self.criterion(y, y_)

                ##Backward pass
                if np.isfinite(loss.data[0]):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            if epoch % self.config['test_interval'] == 0:
                self.run_validation(self.val_dataset)

    def run_validation(self, loader_val):
        for step, data in enumerate(loader_val):
            ##Prepare data
            inputs, targets, input_percentages, target_sizes, _ = data
            inputs = Variable(inputs)
            y_ = Variable(targets)
            if self.device == "cuda":
                inputs = inputs.cuda()
                y_ = y_.cuda()
            ##Forward Pass
            y = self.model(inputs)
            loss = self.criterion(y, y_)
            if loss < self.best_score:
                self.best_score = loss
                self.best_model = copy.deepcopy(self.model)
                print("best model updated, best score: %f" % self.best_score)
                self.save_model_checkpoint(self) 
        self.model.train()
        return           

    def save_model_checkpoint(self):
        torch.save(self.best_model.state_dict(), os.path.join(self.config["model_dir"], f"model_{self.best_score}.pth"))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()