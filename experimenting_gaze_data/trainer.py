
import argparse
import json
from data_processing.data_loader import load_npy_files, GazeDataLoader, FixationDataset
import copy
from experimenting_gaze_data.auto_encoder import gazeAutoEncoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime


    
class Trainer:
    def __init__(self):
        self.args = self.get_arguments()
        config_path = self.args.config_file
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        os.makedirs(self.config["logs_path"], exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = gazeAutoEncoder(latent_dim = self.config['latent_dim'])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.loader_train = self.load_data(self.config["train_data_path"], self.config, "train")
        self.config_val = copy.deepcopy(self.config)
        self.config_val['split_seqs']=False
        self.config_val['batch_size']=1
        self.config_val['augment']=False
        self.loader_val = self.load_data(self.config["val_data_path"], self.config_val, "val")
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
        dataset = FixationDataset(config = self.config, gaze_data = data)
        loader = GazeDataLoader(dataset, batch_size=batch_size,
                                    shuffle=True)
        print(f"{train_val} dataset loaded with {dataset.__len__()} samples")
        return loader

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
                loss = self.criterion(y,inputs)

                ##Backward pass
                if np.isfinite(loss.item()):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            if epoch > self.config['test_interval'] and epoch % self.config['test_interval'] == 0:
                self.run_validation(self.loader_val)

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
            loss = self.criterion(y,inputs)
            if loss < self.best_score:
                self.best_score = loss
                self.best_model = copy.deepcopy(self.model)
                print("best model updated, best score: %f" % self.best_score)
                self.save_model_checkpoint() 
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


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()