#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib


##################################### our code ####################################
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchaudio
import torchaudio.transforms
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
##################################### our code ####################################

def SplitData(root_folder, split=True, split_ratio=0.33):
    patient_ids = find_data_folders(root_folder)

    num_patients = len(patient_ids)

    if num_patients==0:
        raise Exception('No data was provided.')
    
    outcomes = list()
    for i in range(num_patients):
        # Load data.
        patient_id = patient_ids[i]
        patient_metadata_file = os.path.join(root_folder, patient_id, patient_id + '.txt')
        patient_metadata = load_text_file(patient_metadata_file)
        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        current_cpc = get_cpc(patient_metadata)
        outcomes.append([current_outcome, current_cpc])
                
    if split is True:
        X_train, X_test, y_train, y_test = train_test_split(patient_ids, outcomes, 
                                                            test_size=split_ratio, 
                                                            shuffle=True,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        X_train = patient_ids
        y_train = outcomes
        return X_train, y_train
    

class dataset(Dataset):
    def __init__(self, X, y, transforms, sequence, seg=1, train=True):
        self.X = X
        self.y = y
        self.train=train
        self.transforms = transforms
        self.sequence = sequence
        self.seg = seg

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        label = self.y[idx][0]
        cpc = self.y[idx][1]
        #print("cpc: ", cpc)
        #print("label:", label)
        if self.train:
            #x = x[:, 0:h_sequence]
            x = torch.from_numpy(x)
            label = torch.tensor(label)
            cpc = torch.tensor(cpc, dtype=torch.float)
        else:
            #x = x[:, 0:h_sequence]
            x = torch.from_numpy(x)
            label = torch.tensor(label)
            cpc = torch.tensor(cpc, dtype=torch.float)
        
        #print(x.shape)
        x = self.transforms(x)
        #print(x.shape)
        x = x[:,:,:self.sequence].transpose(1,2)
        #print(x.shape)
        x = x.view(x.shape[0], int(self.sequence/self.seg), self.seg, x.shape[2]) + 0.001
        x = x.log2()
        #print(x.shape)
        label = torch.full((1, x.shape[2]), label, dtype=torch.long).squeeze()
        cpc = torch.full((1, x.shape[2]), cpc, dtype=torch.long).squeeze()
        #print(label.shape)
        #print(label)
        return {"data":x, "label":label, "cpc": cpc/5}

class PositionEncoder(nn.Module):
    def __init__(self, n_dim, dropout=0.5, s=1000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(0, s, dtype=torch.float).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, n_dim, 2, dtype=torch.float) / n_dim)
        pe = torch.zeros([s, n_dim])
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)  
    
class Transformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nclass, dropout=0.5, mask_s=0):
        super(Transformer, self).__init__()
        self.ninp = ninp
        self.encoder = nn.Sequential(nn.Linear(ntoken, ninp),
                                     nn.Tanh())
        self.pos_encoder = PositionEncoder(ninp, dropout)
        encoderlayer = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoderlayer, nlayers)
        self.decoder = nn.Linear(ninp, nclass)
        self.regressor = nn.Linear(ninp, 1)
        self.init_weights()
        
        if(mask_s is not 0):
            mask = (torch.triu(torch.ones(mask_s, mask_s), diagonal=1) == 1)
            mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
            self.register_buffer("mask", mask)
        else:
            self.mask  = None
            
    def init_weights(self):
        initrange = 0.1
        self.encoder[0].bias.data.zero_()
        self.encoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.regressor.bias.data.zero_()
        self.regressor.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):  
        x = x[0,:,:,:]
        #print(x.shape)     
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = self.pos_encoder(x)
        #print(x.shape)
        #print("="*10) 
        x = self.transformer_encoder(x, self.mask) if self.mask is not None else self.transformer_encoder(x)
        #print(x.shape)
        cpc_pred = self.regressor(x[x.shape[0]-1,:,:])
        class_pred = self.decoder(x[x.shape[0]-1,:,:])
        return class_pred, cpc_pred

def load_train_test_data(root_folder, channels, X, y):
    # Extract features from the recording data and metadata.
    available_signal_data = list()
    available_labels = list()
    num_channels = len(channels)
    for i in range(len(X)):
        patient_id = X[i]
        patient_label = y[i]
        #print(patient_label)
        patient_metadata, recording_metadata, recording_data = load_challenge_data(root_folder, patient_id)
        num_recordings = len(recording_data)
        for j in range(num_recordings):
            signal_data, sampling_frequency, signal_channels = recording_data[j]
            if signal_data is not None:
                signal_data = reorder_recording_channels(signal_data, signal_channels, channels)
                if signal_data.sum() != 0.0:
                    available_signal_data.append(signal_data)
                    available_labels.append(patient_label)
                    
    return available_signal_data, available_labels

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    ################################# Challenge Models ################################
    ################################# Challenge Models ################################

    ################################# Our Models ################################
    split = False
    ratio = 0.05
    #X_train_, X_test_, y_train_, y_test_ = SplitData(data_folder, split=split, split_ratio=ratio)
    X_train_, y_train_ = SplitData(data_folder, split=split, split_ratio=ratio)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
            'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    
    X_train, y_train = load_train_test_data(data_folder, channels, X_train_, y_train_)
    #X_test, y_test = load_train_test_data(data_folder, channels, X_test_, y_test_)
    #X_train = X_test
    #y_train = y_test
    transforms = torchaudio.transforms.MelSpectrogram(sample_rate=100, n_fft=200, n_mels=128)

    #X_test_, y_test_ = Dataset(X_test, y_test, transforms)
    
    #hyper parameters
    h_sequence = 300
    h_seg = 30

    #learing parameters
    h_lr = 0.0001
    h_stepsize = 2
    h_decay = 0.95
    h_epoch = 150
    h_test = 10

    # transformer
    ntoken = 128 # input dim
    ninp = 512 # embedding dim
    nhead = 8 # head num
    nhid = 2048 # hidden layer dim
    nlayers = 8 # Nx
    nclass = 2 # output dim

    train_data = dataset(X_train, y_train, transforms, h_sequence, h_seg)
    outcome_model = Transformer(ntoken, ninp, nhead, nhid, nlayers, nclass, 0.3, int(h_sequence/h_seg)).cuda()

    loss_fn = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()
    optim = torch.optim.SGD(outcome_model.parameters(), lr=h_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, h_stepsize, h_decay)

    try:
        for epoch in range(h_epoch):
            """
            Train
            """
            total_loss = 0.0
            TP = 0 
            FP = 0
            outcome_model.train()
            for i, data in enumerate(train_data):
                optim.zero_grad()
                inputs, labels, cpcs = data["data"].cuda(), data["label"].cuda(), data["cpc"].cuda()
                outputs, cpc_preds = outcome_model(inputs)
                cpc_preds = cpc_preds.squeeze()
                loss1 = loss_fn(outputs, labels)
                loss2 = loss_reg(cpcs, cpc_preds)
                loss = loss1 + loss2
                loss.backward()
                optim.step()
                
                total_loss += loss.item()
                predict = labels == torch.argmax(outputs, dim=1)
                TP += (predict == True).sum()
                FP += (predict == False).sum()
                
            precision = TP / float(TP+FP) * 100

            print("[{:3d}/{:3d}] loss: {:.6f}".format(epoch+1, h_epoch, total_loss))
            
            """
            Test
            """
            if(epoch % h_test == h_test-1):
                total_loss = 0.0
                TP = 0 
                FP = 0
                outcome_model.eval()
                for i , data in enumerate(train_data):
                    inputs, labels, cpcs = data["data"].cuda(), data["label"].cuda(), data["cpc"].cuda()
                    outputs, cpc_preds = outcome_model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss2 = loss_reg(cpcs, cpc_preds)
                    loss = loss1 + loss2
                    total_loss += loss.item()
                    
                    predict = labels == torch.argmax(outputs, dim=1)
                    TP += (predict == True).sum()
                    FP += (predict == False).sum()
                    
                precision = TP / float(TP+FP) * 100
                print("----- Test -----")
                print("Precision: {:5.2f}%".format(precision))
                print("-----------------")
                    
            scheduler.step()

    except KeyboardInterrupt:
        print('Ctrl+C, saving snapshot')
        # Save the models.
        save_challenge_model(model_folder, outcome_model)
        print('done.')
        return
    ################################# Our Models ################################

    # Save the models.
    save_challenge_model(model_folder, outcome_model)

    if verbose >= 1:
        print('Done.')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def original_load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.pt')
    return torch.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    #imputer = models['imputer']
    #outcome_model = models['outcome_model']
    #cpc_model = models['cpc_model']

    # Load data.
    #patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
            'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    
    Test_Data = []
    num_channels = len(channels)
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
    num_recordings = len(recording_data)
    for j in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[j]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels)
            if signal_data.sum() != 0.0:
                Test_Data.append(signal_data)
    try:
        X_test = Test_Data[0]
        X_test = torch.from_numpy(X_test)
        transforms = torchaudio.transforms.MelSpectrogram(sample_rate=100, n_fft=200, n_mels=128)
        #hyper parameters
        h_sequence = 300
        h_seg = 30
        #print(x.shape)
        x = transforms(X_test)
        #print(x.shape)
        x = x[:,:,:h_sequence].transpose(1,2)
        #print(x.shape)
        x = x.view(x.shape[0], int(h_sequence/h_seg), h_seg, x.shape[2]) + 0.001
        x = x.log2()
        x = x.cuda()

        # Apply models to features.
        outputs, cpc = models(x)
        outcome_probabilities = F.softmax(outputs[0,:], dim=0)
        outcome = torch.argmax(outcome_probabilities)
        outcome_probability = outcome_probabilities[outcome]
        outcome_probability = outcome_probability.data.cpu().item()
        outcome = outcome.data.cpu().item()
        
        #outcome = outcome_model.predict(features)[0]
        #outcome_probability = outcome_model.predict_proba(features)[0, 1]
        #cpc = cpc_model.predict(features)[0]
        # Ensure that the CPC score is between (or equal to) 1 and 5.
        cpc = cpc*5 #np.clip(cpc, 1, 5)
        cpc = cpc.mean().data.cpu().item()
    except:
        outcome, outcome_probability, cpc = float('nan'), float('nan'), float('nan')
        
    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, outcome_model): #, imputer, outcome_model, cpc_model):
    torch.save( outcome_model, os.path.join(model_folder, 'models.pt'))
    #d = {'imputer': imputer, 'cpc_model': cpc_model}
    #filename = os.path.join(model_folder, 'models.sav')
    #joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
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

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features
