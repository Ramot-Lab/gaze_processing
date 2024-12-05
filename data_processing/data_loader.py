import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
from constants import *
from utils import BoxMuller_gaussian
import matplotlib.pyplot as plt

def load_npy_files(data_path, config, train_val='train'):
    
    """
    Loads npy files, filters them according to config['filter'][train_val] and config['events'],
    and returns a list of structured arrays with the required fields.

    Parameters
    ----------
    data_path : str
        loader data location
    config : dict
        Dictionary containing the configuration.
    train_val : str, optional
        Whether to filter for 'train' or 'val'. Default is 'train'.

    Returns
    -------
    data_list : list
        List of structured arrays with the required fields.
    """
    FILES = glob.glob(os.path.join(data_path, "**",'*.npy'), recursive=True)
    FILES = filter_relevant_files(FILES, config, train_val)
    data_list = []
    for _fpath in FILES:
        _data = np.load(_fpath)
        _data[:,-1][_data[:,-1]==0] = 1 # 0 notations are for seperation between annotation secssions (when annotated with the model) setting them to fixations for smoothness 
        if _data.dtype.names is None or not set(_data.dtype.names).issuperset({'t', 'x', 'y', 'evt', 'status'}):
            _data = np.core.records.fromarrays([_data[:,0], _data[:,1], _data[:,2], _data[:,3], _data[:,4]],
                                               dtype=[('t', '<f8'), ('x', '<f4'), ('y', '<f4'), ('status', '<u1'), ('evt', '<u1')])
        _mask = np.in1d(_data['evt'], config['events'])
        _data['status'][~_mask] = False

        data_list.append(_data)
    return data_list

def filter_relevant_files(npy_files, config, train_val='train'):
    """
    Filter npy files based on config['filter'][train_val].

    Parameters
    ----------
    npy_files : list
        List of npy files to filter.
    config : dict
        Dictionary containing the filter configuration.
    train_val : str, optional
        Whether to filter for 'train' or 'val'. Default is 'train'.

    Returns
    -------
    filtered_files : list
        List of filtered npy files.
    """
    return [f for f in npy_files if os.path.split(f)[-1].split("_")[0] in config['filter'][train_val]]

class EventParser(object):
    def __init__(self, config):
        """

        """
        super(EventParser, self).__init__()
        self.config = config


    def parse_data(self, sample):
        config = self.config
        augment = config['augment']
        rms_noise_levels = np.arange(*config["augment_noise"])

        inpt_dir = ['x', 'y']

        gaze_x = np.copy(sample[inpt_dir[0]])
        gaze_y = np.copy(sample[inpt_dir[1]])

        if augment:
            u1, u2 = np.random.uniform(0,1, (2, len(sample)))
            noise_x, noise_y = BoxMuller_gaussian(u1,u2)
            rms_noise_level = np.random.choice(rms_noise_levels)
            noise_x*=rms_noise_level/2
            noise_y*=rms_noise_level/2
            #rms = np.sqrt(np.mean(np.hypot(np.diff(noise_x), np.diff(noise_y))**2))
            gaze_x+=noise_x
            gaze_y+=noise_y

        inpt_x, inpt_y = [np.diff(gaze_x),
                          np.diff(gaze_y)]

        X = [(_coords) for _coords in zip(inpt_x, inpt_y)]
        X = np.array(X, dtype=np.float32)

        return X


class   EMDataset(Dataset, EventParser):
    def __init__(self, config, gaze_data):
        """
        Dataset that loads tensors
        """

        split_seqs = config['split_seqs']
        #mode = config['mode']
        center_data = config['center_data']
        stride_step = config['stride_step']
        #input is in fact diff(input), therefore we want +1 sample
        seq_len = config['seq_len']+1

        data = []
        #seqid = -1
        for d in gaze_data: #iterates over files
            d = self.data_preprocess(d, normalize=center_data)
            dd = np.split(d, np.where(np.diff(d['status'].astype(np.int0)) != 0)[0]+1)
            dd = [_d for _d in dd if (_d['status'].all() and not(len(_d) < seq_len))]

            for seq in dd: #iterates over chunks of valid data
                #seqid +=1
                if split_seqs and not(len(seq) < seq_len):
                    #this
                    #1. overlaps the last piece of data and
                    #2. allows for overlaping sequences in general; not tested
                    seqs = [seq[pos:pos + seq_len] if (pos + seq_len) < len(seq) else
                            seq[len(seq)-seq_len:len(seq)] for pos in range(0, len(seq), stride_step)]
                else:
                    seqs = [seq]

                data.extend(seqs)

        self.data = data
        self.size = len(data)
        self.config = config

        super(EMDataset, self).__init__(config)

    def __getitem__(self, index):
        sample = self.data[index]
        gaze_data = self.parse_data(sample)
        evt = self.parse_evt(sample['evt'])

        return torch.FloatTensor(gaze_data.T), evt, ()

    def parse_evt(self, evt):
        return evt[1:]-1

    def __len__(self):
        return self.size

    def data_preprocess(self, sample, normalize=True):  
        if normalize:
            sample[:,1:3] = sample[:,1:3] - sample[:,1:3].mean(axis=0)
        return sample

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    #return batch
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = [] #torch.IntTensor(minibatch_size, 3)
    targets = []

    for x in range(minibatch_size):
        sample = batch[x]

        tensor, target, (_) = sample
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes.append(len(target))
        targets.extend(target.tolist())
    targets = torch.LongTensor(targets)
    return inputs, targets, input_percentages, target_sizes, (_)



class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        torch.manual_seed(220617)
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)

class GazeDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader
        """
        seed = kwargs.pop('seed', 220617)
        super(GazeDataLoader, self).__init__(*args, **kwargs)
        np.random.seed(seed)
        self.collate_fn = _collate_fn
        #self.sampler = RandomSampler(*args)


class FixationDataset(EMDataset):
    
    def __init__(self, config, gaze_data):
        self.config = config
        gaze_data = self.break_data_into_fixaitons(gaze_data)
        super().__init__(config, gaze_data)

    def break_data_into_fixaitons(self, gaze_data):
        fixation_gaze_data = []
        for recording in gaze_data:
            recording = recording[recording['evt'] == FIXATION_IDX]
            separation_parts = np.where(np.diff(recording['t'])>np.median(np.diff(recording['t'])+2))[0]
            i = 0
            for j in separation_parts:
                if j - i < self.config['seq_len']: 
                    i = j
                    continue
                fixation_gaze_data.append(recording[i:j+1])
                i = j+1
        return fixation_gaze_data
