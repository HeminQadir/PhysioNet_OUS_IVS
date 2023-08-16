#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%%
# Libraries for the first entery
from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm, MSELoss

#%%
#libraries for the second entery
import ml_collections
import math
import copy
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


#%%
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self): 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#%%
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


#%%
def set_seed(seed=42, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


#%%
def load_train_val_files(data_folder, split=True, split_ratio=0.1):

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    if split:
        X_train, X_val = train_test_split(patient_ids, test_size=split_ratio, 
                                        shuffle=True, random_state=42)
        return X_train, X_val
    else:
        X_train = patient_ids
        return X_train



#%%
def rescale_data(data):
    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
    return data 

#%%
def get_labels(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    # Extract labels.
    outcome = int(get_outcome(patient_metadata))

    outcome = torch.tensor(outcome, dtype=torch.long)
    return outcome

#%%
# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm), dtype=np.float32)
    return features

#%%
# Save your trained model.
def save_challenge_model(model_folder, outcome_model, epoch): #, imputer, outcome_model, cpc_model):
    torch.save({'model': outcome_model, 'epoch': epoch,}, os.path.join(model_folder, 'model.pt'))

def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.pt')
    state = torch.load(filename)
    model = state['model']
    return model 

#%%
def number_to_one_hot(number, num_classes):
    identity_matrix = torch.eye(num_classes)
    one_hot = identity_matrix[number]
    return one_hot

#%%
# Load the WFDB data for the Challenge (but not all possible WFDB files).
def load_recording_header(record_name, check_values=True):
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext=='':
        header_file = record_name + '.hea'
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError('{} recording not found.'.format(record_name))

    with open(header_file, 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]

    # Parse the header file.
    record_name = None
    sampling_frequency = None
    length = None 

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(' ')]
        # Parse the record line.
        if i==0:
            record_name = arrs[0]
            sampling_frequency = float(arrs[2])
            length = int(arrs[3])

    return sampling_frequency, length

#%%
# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm), dtype=np.float32)

    return features


#%%
# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 100
    else:
        resampling_frequency = 100
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    return data, resampling_frequency


#%%
def rescale_data(data):
    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
    return data 


#%%
def get_labels(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    # Extract labels.
    outcome = int(get_outcome(patient_metadata))

    outcome = torch.tensor(outcome, dtype=torch.long)
    return outcome


#%%
def segment_eeg_signal(eeg_signal, window_size, step_size, Fs):
    """
    window_size ---> in min
    window_step ---> in min
    """
    if Fs == 0:
        Fs = 100

    # Calculate the number of samples per window
    window_samples = int(round(window_size*60*Fs))

    # Calculate the number of samples to move the window by
    overlap_samples = int(step_size*60*Fs)

    # Determine the total number of windows
    num_windows = int(np.ceil((eeg_signal.shape[1] - window_samples) / overlap_samples)) + 1

    # Segment the EEG signal into windows
    segments = []
    for i in range(num_windows):
        start_index = i * overlap_samples
        end_index = start_index + window_samples
        
        if end_index > eeg_signal.shape[1]:
            break
        window = eeg_signal[:, start_index:end_index]
        segments.append(window)
    return segments



#%%
def load_data(data_folder, patient_id, train=True):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)
    # Handle NaN value
    patient_features[patient_features != patient_features] = 0.0
    #print(patient_features)

    # Load EEG recording.    
    eeg_channels = ['F3', 'P3', 'F4', 'P4'] #['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2'] # 
    group = 'EEG'

    size = 30000
    bipolar_data = np.zeros((2, size), dtype=np.float32)
    # bipolar_data = np.zeros((18, data.shape[1]), dtype=np.float32)

    # check if there is at least one EEG record
    if num_recordings > 0:
        random.shuffle(recording_ids)
        
        for recording_id in recording_ids:    #for recording_id in reversed(recording_ids):
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                sampling_frequency, length = load_recording_header(recording_location, check_values=True)  # we created to read only the header and get the fs
                five_min_recording = sampling_frequency * 60 * 5

                # checking the length of the hour recording 
                if length >= five_min_recording:
                    data, channels, sampling_frequency = load_recording_data(recording_location, check_values=True)
                    utility_frequency = get_utility_frequency(recording_location + '.hea')

                    # checking if we have all the channels 
                    if all(channel in channels for channel in eeg_channels):
                        data, channels = reduce_channels(data, channels, eeg_channels)
                        data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                        data = rescale_data(data)
                        bipolar_data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]], dtype=np.float32) # Convert to bipolar montage: F3-P3 and F4-P4 
                        
                        # bipolar_data = np.zeros((18, data.shape[1]), dtype=np.float32)

                        # bipolar_data[8,:] = data[0,:] - data[1,:];     # Fp1-F3
                        # bipolar_data[9,:] = data[1,:] - data[2,:];     # F3-C3
                        # bipolar_data[10,:] = data[2,:] - data[3,:];    # C3-P3
                        # bipolar_data[11,:] = data[3,:] - data[7,:];    # P3-O1
                    
                        # bipolar_data[12,:] = data[11,:] - data[12,:];  # Fp2-F4
                        # bipolar_data[13,:] = data[12,:] - data[13,:];  # F4-C4
                        # bipolar_data[14,:] = data[13,:] - data[14,:];  # C4-P4
                        # bipolar_data[15,:] = data[14,:] - data[18,:];  # P4-O2
                    
                        # bipolar_data[0,:] = data[0,:] - data[4,:];     # Fp1-F7
                        # bipolar_data[1,:] = data[4,:] - data[5,:];     # F7-T3
                        # bipolar_data[2,:] = data[5,:] - data[6,:];     # T3-T5
                        # bipolar_data[3,:] = data[6,:] - data[7,:];     # T5-O1
                    
                        # bipolar_data[4,:] = data[11,:] - data[15,:];   # Fp2-F8
                        # bipolar_data[5,:] = data[15,:] - data[16,:];   # F8-T4
                        # bipolar_data[6,:] = data[16,:] - data[17,:];   # T4-T6
                        # bipolar_data[7,:] = data[17,:] - data[18,:];   # T6-O2
                    
                        # bipolar_data[16,:] = data[8,:] - data[9,:];    # Fz-Cz
                        # bipolar_data[17,:] = data[9,:] - data[10,:];   # Cz-Pz

                        break

                    else:
                        pass
                else:
                    pass
            else: 
                pass

    #last_5_min = int(sampling_frequency * 60 * 5)
    #last_5_min_data = bipolar_data[:, -last_5_min:]
    
    sampling_frequency = 100
    segments = segment_eeg_signal(bipolar_data, 5, 3, sampling_frequency)
    indx = random.randint(0, len(segments)-1)
    data_5_min = segments[indx]

    #last_5_min_data = rescale_data(last_5_min_data)
    
    if train:
        # Extract labels.
        outcome = int(get_outcome(patient_metadata))
        #print(outcome)
        cpc = int(get_cpc(patient_metadata))
        #print(cpc)

        x = torch.from_numpy(data_5_min)
        outcome = torch.tensor(outcome, dtype=torch.long)
        cpc = torch.tensor(cpc, dtype=torch.long)

        #outcome = number_to_one_hot(outcome, 2)

        return x, outcome, cpc
    
    else:
        x = torch.from_numpy(data_5_min)

        return x


#%%
class dataset(Dataset):
    def __init__(self, data_folder, X_files, train=True):
        self.X_files = X_files
        self.train=train
        self.data_folder = data_folder

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        patient_id = self.X_files[idx]

        x, outcome, cpc = load_data(self.data_folder, patient_id)
        
        return {"input":x, "outcome":outcome, "cpc": cpc/5.0}


#%%
class targets(Dataset):
    def __init__(self, data_folder, X_files, train=True):
        self.X_files = X_files
        self.train=train
        self.data_folder = data_folder

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        patient_id = self.X_files[idx]

        outcome = get_labels(self.data_folder, patient_id)
        
        return {"outcome":outcome}


#%%
def get_class_weights(targets):
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (2 * class_counts)
    return class_weights


#%%
def get_upsampled_loader(targets):
    class_weights = get_class_weights(targets)
    samples_weights = class_weights[targets]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    return sampler


def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': 1000})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 8
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

#%%
def setup(input_length, num_classes, in_channels, device):
    # Prepare model
    config =  get_config()
    model = VisionTransformer(config, input_length, zero_head=True, num_classes=num_classes)
    model.to(device)
    num_params = count_parameters(model)
    print(model)
    print(num_params)
    return model

# Save your trained model.
def save_challenge_model(model_folder, outcome_model, epoch): #, imputer, outcome_model, cpc_model):
    torch.save({'model': outcome_model, 'epoch': epoch,}, os.path.join(model_folder, 'model.pt'))

def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.pt')
    state = torch.load(filename)
    model = state['model']
    return model 

#%%
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

#%%
def valid(model, val_loader, global_step, eval_batch_size, local_rank, device):
    # Validation!
    eval_losses = AverageMeter()


    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for step, batch in enumerate(epoch_iterator):

        data = batch
        x, y, cpcs = data["input"].to(device), data["outcome"].to(device), data["cpc"].to(device)
    
        with torch.no_grad():
            logits, regression, _ = model(x) #[0]   #[0] not needed if we have only one output

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            #print(preds)

            predict = y == preds

            TP += (predict == True).sum()
            FP += (predict == False).sum()
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    #accuracy = simple_accuracy(all_preds, all_label)
    precision = TP / float(TP+FP) * 100

    print("precision: ", precision)

    return precision #accuracy


#%%
def train(model, data_folder, model_folder, device, num_steps, eval_every, local_rank, train_batch_size, eval_batch_size, learning_rate, n_gpu):
    """ Train the model """
    name = "physionet"
    gradient_accumulation_steps = 1
    decay_type = "cosine" #choices=["cosine", "linear"]
    warmup_steps = 500 
    max_grad_norm = 1.0

    if local_rank in [-1, 0]:
        os.makedirs(model_folder, exist_ok=True)


    train_batch_size = train_batch_size // gradient_accumulation_steps

    split = True
    split_ratio = 0.1
    shuffle = True

    if split:
        X_train, X_val = load_train_val_files(data_folder, split, split_ratio)
        
        trainset = dataset(data_folder, X_train)
        label = targets(data_folder, X_train)
        train_labels = list()
        for i, data in enumerate(label):
            label = data["outcome"]
            train_labels.append(label.item())
        train_labels = torch.tensor(train_labels)
        sampler = get_upsampled_loader(train_labels)
        train_loader =  DataLoader(trainset, batch_size=train_batch_size, sampler=sampler) #shuffle=shuffle)

        valset = dataset(data_folder, X_val)
        val_loader =  DataLoader(valset, batch_size=eval_batch_size, shuffle=shuffle)


    else:
        X_train = load_train_val_files(data_folder, split, split_ratio)
        trainset = dataset(data_folder, X_train)
        label = targets(data_folder, X_train)
        train_labels = list()
        for i, data in enumerate(label):
            label = data["outcome"]
            train_labels.append(label.item())
        train_labels = torch.tensor(train_labels)
        sampler = get_upsampled_loader(train_labels)
        train_loader =  DataLoader(trainset, batch_size=train_batch_size, sampler=sampler) #shuffle=shuffle)

        #data_folder_val = "/media/jacobo/NewDrive/physionet.org/files/i-care/2.0/validation"
        #X_val = load_train_val_files(data_folder_val, split, split_ratio)
        #valset = dataset(data_folder_val, X_val)
        #val_loader =  DataLoader(valset, batch_size=eval_batch_size, shuffle=shuffle)
    

    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                betas=(0.9, 0.999), 
                                eps=1e-08,
                                weight_decay=weight_decay)

    t_total = num_steps
    if decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)


    # Train!
    model.zero_grad()
    seed = 42
    set_seed(seed=seed, n_gpu=n_gpu)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            data = batch
            x, y, cpcs = data["input"].to(device), data["outcome"].to(device), data["cpc"].to(device)
            #print("I am label: ", y)
            loss1, loss2 = model(x, y, cpcs)

            loss = loss1 + loss2 

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                losses.update(loss.item()*gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) (loss_class=%2.5f) (loss_regress=%2.5f)" % (global_step, t_total, losses.val, loss1, loss2)
                )

                if global_step % eval_every == 0 and local_rank in [-1, 0]:
                    accuracy = valid(model, val_loader, global_step, eval_batch_size, local_rank, device)
                    if best_acc < accuracy:
                        #save_challenge_model(args, model)
                        save_challenge_model(model_folder, model, global_step)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break


def train_challenge_model(data_folder, model_folder, verbose=2):
    # Required parameters
    
    input_length = 30000
    train_batch_size = 10
    eval_batch_size = 10 
    eval_every = 500
    learning_rate = 1e-4 
    num_steps = 20000
    local_rank = -1
    seed = 42
    fp16 = False
    num_classes = 2
    in_channels = 2

    # Setup CUDA, GPU & distributed training
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        n_gpu = 1


    # Set seed
    set_seed(seed=seed, n_gpu=n_gpu)

    # Model & Tokenizer Setup
    model = setup(input_length, num_classes, in_channels, device)

    # Training
    train(model, data_folder, model_folder, device, num_steps, eval_every, local_rank, train_batch_size, eval_batch_size, learning_rate, n_gpu)


#%%
# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    x = load_data(data_folder, patient_id, train=False)
    x = x.cuda()

    if len(x)>0:
        # Apply models to features.
        models.eval()
        outputs, pred_cpcs, _ = models(x.unsqueeze(0))
        
        outcome_probabilities = F.softmax(outputs[0])
                
        pred_outcome = torch.argmax(outcome_probabilities)
        outcome_probability = outcome_probabilities[1]   # predicted probability of a poor outcome
        outcome_probability = outcome_probability.data.cpu().item()
        pred_outcome = pred_outcome.data.cpu().item()

        pred_cpcs = pred_cpcs*5
        pred_cpcs = pred_cpcs.data.cpu().item()
        pred_cpcs = np.clip(pred_cpcs, 1, 5)  

        print("="*80)
        print(pred_outcome)
        print(pred_cpcs)
        #outcome_probability = round(outcome_probability, 2)
        print(outcome_probability)
        
    else:
        pred_outcome, outcome_probability, pred_cpcs = float(0), float(0), float(0) #float('nan'), float('nan'), float('nan')

    return pred_outcome, outcome_probability, pred_cpcs


#%%
# The models 

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        #print("I am in the attention head")
        #print(hidden_states.shape)
        #torch.Size([10, 25, 768])

        mixed_query_layer = self.query(hidden_states)
        #print(mixed_query_layer.shape)
        #torch.Size([10, 25, 768])

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        #print(query_layer.shape)
        #torch.Size([10, 2, 25, 384])

        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        #print(attention_output.shape)
        #torch.Size([10, 25, 768])
        #print("End of the attention head")
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        #print("I am in the MLP")
        x = self.fc1(x)
        #print(x.shape)
        x = self.act_fn(x)
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(x.shape)
        x = self.dropout(x)
        #print("End of the MLP")
        return x 


# Define your deep learning model
class Embeddings(nn.Module):
    def __init__(self, config, input_length, in_channels=2):
        super(Embeddings, self).__init__()
        self.hidden_size = config.hidden_size
        self.in_channels = in_channels
        patch_size = config.patches["size"]

        # self.patch_embeddings = nn.Conv1d(in_channels=in_channels, 
        #                                   out_channels=config.hidden_size*in_channels, 
        #                                   kernel_size=patch_size, 
        #                                   stride=patch_size, 
        #                                   groups=in_channels)

        self.patch_embeddings =  FeatureExtractor(config, in_channels)
        #torch.Size([10, 512, 12])

        n_patches = 24 #in_channels*(int((input_length - patch_size) / patch_size ) + 1)
        #print(n_patches)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        #print("I am in the Embeding Layer")
        #print(x.shape)
        #torch.Size([10, 2, 30000])
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #print(cls_tokens.shape)
        #torch.Size([10, 1, 768])

        x = self.patch_embeddings(x)
        #print(x.shape)
        #torch.Size([10, 1536, 12])

        x = x.view(x.shape[0], self.in_channels, self.hidden_size, x.shape[2])
        #print(x.shape)
        #torch.Size([10, 2, 768, 12])

        x = x.transpose(-2, -3)
        #print(x.shape)
        #torch.Size([10, 768, 2, 12])

        x = x.flatten(2)
        #print(x.shape)
        #torch.Size([10, 768, 24])

        x = x.transpose(-1, -2)
        #print(x.shape)
        #torch.Size([10, 24, 768])

        x = torch.cat((cls_tokens, x), dim=1)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        #embeddings = self.position_embeddings(x)
        k = self.position_embeddings
        #print(k.shape)
        #torch.Size([1, 25, 768])

        embeddings = x + k
        #print(embeddings.shape)
        #torch.Size([10, 25, 768])
        embeddings = self.dropout(embeddings)
        #print("End of the Embeding Layer")

        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        #print("I am in the Block Lalyer")
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = self.attention_norm(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x, weights = self.attn(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = x + h
        #print(x.shape)
        #torch.Size([10, 25, 768])

        h = x
        x = self.ffn_norm(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = self.ffn(x)
        #print(x.shape)
        #torch.Size([10, 25, 768])

        x = x + h
        #print(x.shape)
        #torch.Size([10, 25, 768])
        #print("End of the Block Lalyer")
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config,  input_length, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, input_length)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, input_length=30000, num_classes=1000, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, input_length, vis)
        self.head = Linear(config.hidden_size*25, num_classes)
        self.regress = Linear(config.hidden_size*25, 1)

    def forward(self, x, labels=None, cpcs=None):
        #print(x.shape)
        x, attn_weights = self.transformer(x)
        #print('I am in the VisionTransformer')
        #print(x.shape)
        #torch.Size([1, n_takens, 768])
        #print(x[:, 0].shape)
        #torch.Size([1, 768])
        x = x.flatten(1)
        #print(x.shape)
        logits = self.head(x) #[:, 0])
        regression = self.regress(x) #[:, 0])
        #torch.Size([1, 1000])
        #print(logits.shape)
        #print("I am output: ", logits)
        #print('End of the VisionTransformer')

        if labels is not None and cpcs is not None:
            # if batch size = 9
            #print("I am in the loss")
            #print(labels.view(-1))
            #print(logits.view(-1, self.num_classes).shape)
            loss_fct = CrossEntropyLoss()
            l2_loss = MSELoss()
            loss1 = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            loss2 = l2_loss(regression.squeeze(1), cpcs)
            #loss = loss1 + loss2
            #print(loss)
            return loss1, loss2 
        else:
            return logits, regression, attn_weights

class GroupNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.activation = nn.GELU()
        self.layer_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-05, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x
        
class NoLayerNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()

        out_channels  = int((config.hidden_size)*in_channels)

        self.conv_layers = nn.ModuleList([
            GroupNormConvLayer(in_channels, out_channels, 10, 5, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 5, 2, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 3, groups=in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 3, groups=in_channels)
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
    
class FeatureProjection(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()

        out_channels  = int((config.hidden_size/2)*in_channels)

        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-05)
        self.projection = nn.Linear(out_channels, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

######################################## 1D CNN Classifier ##########################################
class Classification_1DCNN(nn.Module):
    def __init__(self, config, num_classes, in_channels):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(config, in_channels)
        self.feature_projection = FeatureProjection(config, in_channels)
        
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
        return outputs

    def forward(self, inputs, labels=None, **kwargs):
        #print("="*10)
        #print(inputs.shape)
        #torch.Size([5, 660000])
        
        x = inputs #.unsqueeze(1)
        #print(x.shape)
        #torch.Size([5, 1, 660000])
        
        x = self.feature_extractor(x)
        #print(x.shape)
        #torch.Size([5, 512, 2062])
        
        x = x.transpose(1, 2)
        #print(x.shape)
        #torch.Size([5, 2062, 512])
        
        x = self.feature_projection(x)
        #print(x.shape)
        #torch.Size([5, 2062, 768])
        
        x = self.merged_strategy(x, mode="mean")
        #print(merged_features_proj.shape)
        #torch.Size([5, 768])
        
        logits = self.classifier(x)
        #print(logits.shape)
        #torch.Size([5, 10])

        if labels is not None:
            # if batch size = 9
            #print("I am in the loss")
            #print(labels.view(-1))
            #print(logits.view(-1, self.num_classes).shape)
            #loss_fct = BCEWithLogitsLoss()
            #loss =  loss_fct(logits.squeeze(), labels.float())#
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1)) #loss_fct(logits, labels) #
            #print(loss)
            return loss 
        else:
            return logits
