#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
from torch.nn import functional as F

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

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
def shuffle(idx):
    random.shuffle(idx)  # Shuffle the indices
#%%
class dataset(Dataset):
    def __init__(self, data_folder, X_files, train=True):
        self.X_files = X_files
        self.train=train
        self.data_folder = data_folder
        #self.indices = list(range(len(X_files))) 
        #shuffle(self.indices)

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        patient_id = self.X_files[idx]

        #shuffled_index = self.indices[idx]
        #patient_id = self.X_files[shuffled_index]

        patient_metadata = load_challenge_data(self.data_folder, patient_id)
        # Extract patient features.
        patient_features = get_patient_features(patient_metadata)
        #print(patient_features)
        patient_features[patient_features != patient_features] = 0.0

        # Extract labels.
        outcome = int(get_outcome(patient_metadata))
        #print(outcome)
        cpc = int(get_cpc(patient_metadata))
        #print(cpc)

        x = torch.from_numpy(patient_features)
        outcome = torch.tensor(outcome, dtype=torch.long)
        cpc = torch.tensor(cpc, dtype=torch.long)
        
        return {"input":x, "outcome":outcome, "cpc": cpc/5.0}

#%%
# Define your deep learning model
class Two_Layers_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, nclass):
        super(Two_Layers_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, nclass)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#%%

class Multi_layers_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, nclass):
        super(Multi_layers_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.classifier = nn.Linear(hidden_size, nclass)
        self.regressor = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        output = self.classifier(x)
        cpc = self.regressor(x)
        return output, cpc 

    
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
# %%
# Save your trained model.
def save_challenge_model(model_folder, outcome_model, epoch): #, imputer, outcome_model, cpc_model):
    torch.save({'model': outcome_model, 'epoch': epoch,}, os.path.join(model_folder, 'model.pt'))

def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.pt')
    state = torch.load(filename)
    model = state['model']
    return model 

#%%
#data_folder = "/media/jacobo/D"
#model_folder = "/media/jacobo/D/trained_models"

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose=2):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    split_ratio = 0.1
    split = False

    if split:
        X_train, X_val = load_train_val_files(data_folder, split, split_ratio)
        valset = dataset(data_folder, X_val)
        val_data = DataLoader(valset, batch_size=10, shuffle=shuffle)
    else:
        X_train = load_train_val_files(data_folder, split, split_ratio)

    trainset = dataset(data_folder, X_train)
    batch_size = 32
    shuffle = True
    train_data = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    os.makedirs(model_folder, exist_ok=True)

    # Define the input vector
    input_vector = torch.tensor([53., 0., 1., 0., 0., 1., 1., 33.])
    input_vector = input_vector.cuda()
    # Initialize the model
    input_size = len(input_vector)
    hidden_size = 64 # Choose the number of hidden units
    nclass = 2  # Since the output is binary (0 or 1)
    model = Multi_layers_Classifier(input_size, hidden_size, nclass).cuda()
    print(model)

    model_path = Path(os.path.join(model_folder, 'model.pt'))

    if model_path.exists():
        state = torch.load(model_path)
        epoch = state['epoch']
        model = state['model']
        print('Restored model at epoch {}'.format(epoch))
    else:
        print("The model is trained from scratch")
        epoch = 1

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    regression = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100 # Choose the number of training epochs
    h_test = 1

    #old_precision = 0
    try:
        for epoch in range(num_epochs):
            """
            Train
            """
            total_loss = 0.0
            model.train()
            for i, data in enumerate(train_data):
                # Zero the gradients
                optimizer.zero_grad()

                inputs, outcomes, cpcs = data["input"].cuda(), data["outcome"].cuda(), data["cpc"].cuda()
                #print(inputs)
                #print(outcomes)
            
                # Forward pass
                outputs, pred_cpcs = model(inputs)
                #print(outputs)
            
                # Compute the loss
                class_loss = criterion(outputs, outcomes) #(outputs.unsqueeze(0), outcomes.unsqueeze(0))
                regress_loss = regression(cpcs, pred_cpcs)

                loss = class_loss + regress_loss

                # Backward pass
                loss.backward()

                # Update the model parameters
                optimizer.step()

                total_loss += loss.item()
                #print("[{:3d}/{:3d}] loss: {:.6f}".format(epoch+1, num_epochs, loss.item()))

            print("[{:3d}/{:3d}] loss: {:.6f}".format(epoch+1, num_epochs, total_loss))

            """
            Test
            """
            
            if(epoch % h_test == h_test-1) and split:
                TP = 0 
                FP = 0
                cpc_mse_t_list = []
                model.eval()
                for i, data in enumerate(val_data):
                    inputs, outcomes, cpcs = data["input"].cuda(), data["outcome"].cuda(), data["cpc"].cuda()
                    outputs, pred_cpcs = model(inputs)

                    predict = outcomes == torch.argmax(outputs, dim=1)
                    TP += (predict == True).sum()
                    FP += (predict == False).sum()

                    cpc_mse_t = torch.mean((cpcs-pred_cpcs)**2)
                    cpc_mse_t_list.append(cpc_mse_t.item())

                precision = TP / float(TP+FP) * 100
                print("----- Result on Validation -----")
                print("Precision: {:5.2f}%".format(precision))
                print("MSE of CPC: {:5.6f}%".format(sum(cpc_mse_t_list)/len(cpc_mse_t_list)))
                print("--------------------------------")

                # if precision > old_precision:
                #     save_challenge_model(model_folder, outcome_model, epoch)
                #     old_precision = precision

    except KeyboardInterrupt:
        print('Ctrl+C, saving snapshot')
        # Save the models.
        save_challenge_model(model_folder, model, epoch)
        print('The model is saved')

    save_challenge_model(model_folder, model, epoch)

    if verbose >= 1:
        print('Done.')

#%%
# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    patient_metadata = load_challenge_data(data_folder, patient_id)
    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)
    #print(patient_features)
    patient_features[patient_features != patient_features] = 0.0

    # Extract labels.
    #outcome = int(get_outcome(patient_metadata))
    #print(outcome)
    #cpc = int(get_cpc(patient_metadata))
    #print(cpc)

    x = torch.from_numpy(patient_features)
    x = x.cuda()

    #outcome = torch.tensor(outcome, dtype=torch.long)
    #cpc = torch.tensor(cpc, dtype=torch.long)

    if len(x)>0:
        # Apply models to features.
        models.eval()
        outputs, cpcs = models(x)
        
        outcome_probabilities = F.softmax(outputs)
                
        outcome = torch.argmax(outcome_probabilities)
        outcome_probability = outcome_probabilities[1]   # predicted probability of a poor outcome
        outcome_probability = outcome_probability.data.cpu().item()
        outcome = outcome.data.cpu().item()

        #cpcs = cpcs*5 
        cpcs = cpcs.data.cpu().item()
        cpcs = np.clip(cpcs, 1, 5)  

        print("="*80)
        print(outcome)
        print(cpcs)
        #outcome_probability = round(outcome_probability, 2)
        print(outcome_probability)
        
    else:
        outcome, outcome_probability, cpcs = float(0), float(0), float(0) #float('nan'), float('nan'), float('nan')

    return outcome, outcome_probability, cpcs

