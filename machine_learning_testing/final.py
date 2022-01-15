################################################################################
# IMPORT
################################################################################

import argparse
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

### torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

### sklearn packages
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

################################################################################
# PARSE ARGUMENTS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("n_epochs", help="Number of epochs to use for training", type=int)
parser.add_argument("n_batches", help="Number of batches to use for training", type=int)
parser.add_argument("r_learn", help="Learning rate for Adam optimiser", type=float)
parser.add_argument("input_dir", help="Directory containing data required")
parser.add_argument("output_dir", help="Directory for data output")
parser.add_argument("model" help="Model to use: lc, cent, sdp, all")
args = parser.parse_args()

################################################################################
# DEFINE CLASSES
################################################################################

class CustomDataLoader(Dataset):

    def __init__(self, filepath, train):

        ### List of global, local, and info files (assumes certain names of files)
        appropriate_directory = "train_lcs/" if train else "val_lcs/"
        self.file_list_local = np.sort(glob.glob(os.path.join(filepath, appropriate_directory+'*-local.npy')))
        self.file_list_global = np.sort(glob.glob(os.path.join(filepath, appropriate_directory+'*-global.npy')))
        self.info_file = os.path.join(filepath, 'collated_scientific_domain_parameters.csv')
        self.tce_info_data = pd.read_csv(self.info_file)

        ### Gather up tce ids
        self.ids = []
        for path in self.file_list_local:
            last_section_parts = (path.split('/')[-1]).split('-')
            self.ids.append(last_section_parts[0] + "-" + last_section_parts[1])
        self.ids = np.array(self.ids)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### Fetch local and global views
        tce_id = self.ids[idx]
        data_local = np.load(self.file_list_local[idx])
        data_global = np.load(self.file_list_global[idx])

        ### Get info file
        tce_data_info = self.tce_info_data[self.tce_info_data["tce_id"]==tce_id]
        pc_class = tce_data_info["pc"].values[0]

        # Remove classification and information columns
        del tce_data_info["pc"]
        del tce_data_info["tic_id"]
        del tce_data_info["tce_id"]

        np_tce_data = tce_data_info.to_numpy()
        np_tce_data.astype(np.float)
        return(data_local[0], data_global[0], data_local[1], data_global[1], np_tce_data[0]), pc_class

class LCModel(nn.Module):
    
    def __init__(self):

        ### Initialise model
        super(TestModel, self).__init__()

        ### Define global convolutional layer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )
        
        ### Define local convolutional layer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### Define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(16576, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_local_cen, x_global_cen, x_star_param):

        ### Get outputs of global and local convolutional layers
        ### N.B. Only using light curves
        out_global = self.fc_global(x_global)
        out_local = self.fc_local(x_local)

        ### Flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### Concatenate global and local views for output layer
        out = torch.cat([out_global, out_local], dim=1)
        out = self.final_layer(out)

        return out


class CentroidModel(nn.Module):
    
    def __init__(self):

        ### Initialise model
        super(TestModel, self).__init__()

        ### Define global convolutional layer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )
        
        ### Define local convolutional layer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### Define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(16576, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_local_cen, x_global_cen, x_star_param):
            
        ### Concatenate light curve and centroid data
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)

        ### Get outputs of global and local convolutional layers
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)

        ### Flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### Concatenate global and local views for output layer
        out = torch.cat([out_global, out_local], dim=1)
        out = self.final_layer(out)

        return out

class ParameterModel(nn.Module):
    
    def __init__(self):

        ### Initialise model
        super(TestModel, self).__init__()

        ### Define global convolutional layer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )
        
        ### Define local convolutional layer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### Define fully connected layer that combines both views and includes stellar parameters
        self.final_layer = nn.Sequential(
            nn.Linear(16592, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_local_cen, x_global_cen, x_star_param):
            
        ### Get outputs of global and local convolutional layers
        ### N.B. Only using light curves, no centroids
        out_global = self.fc_global(x_global)
        out_local = self.fc_local(x_local)

        ### Flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### Concatenate global and local views with stellar parameters
        out = torch.cat([out_global, out_local, x_star_param.squeeze(1)], dim=1)
        out = self.final_layer(out)

        return out

class AllModel(nn.Module):
    
    def __init__(self):

        ### Initialise model
        super(TestModel, self).__init__()

        ### Define global convolutional layer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )
        
        ### Define local convolutional layer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### Define fully connected layer that combines both views and includes stellar parameters
        self.final_layer = nn.Sequential(
            nn.Linear(16592, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_local_cen, x_global_cen, x_star_param):
            
        ### Concatenate light curve and centroid data
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)

        ### Get outputs of global and local convolutional layers
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)

        ### Flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### Concatenate global and local views with stellar parameters
        out = torch.cat([out_global, out_local, x_star_param.squeeze(1)], dim=1)
        out = self.final_layer(out)

        return out

################################################################################
# Define auxillary functions
################################################################################

def train_model(n_epochs, data_loader, val_loader, model, criterion, optimiser):

    ### Empty arrays to fill per-epoch outputs
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_val_ap = []

    ### Loop over number of epochs of training
    for epoch in tqdm(range(n_epochs)):

        ### Loop over batches
        train_loss = torch.zeros(1)
        for x_train_data, y_train in data_loader:

            toi_amount = y_train.sum().item()
            batch_weighting = torch.tensor([toi_amount, (args.n_batches - toi_amount)])/args.n_batches
            ### Get local view, global view and label for training
            x_train_local, x_train_global, x_train_local_cen, x_train_global_cen, x_train_star = x_train_data
            x_train_local = Variable(x_train_local).type(torch.FloatTensor)
            x_train_global = Variable(x_train_global).type(torch.FloatTensor)
            x_train_local_cen = Variable(x_train_local_cen).type(torch.FloatTensor)
            x_train_global_cen = Variable(x_train_global_cen).type(torch.FloatTensor)
            x_train_star = Variable(x_train_star).type(torch.FloatTensor)
            y_train = Variable(y_train).type(torch.FloatTensor)

            ### Fix dimnensions for next steps
            x_train_local = x_train_local.unsqueeze(1)
            x_train_global = x_train_global.unsqueeze(1)
            x_train_local_cen = x_train_local_cen.unsqueeze(1)
            x_train_global_cen = x_train_global_cen.unsqueeze(1)
            x_train_star = x_train_star.unsqueeze(1)
            y_train = y_train.unsqueeze(1)

            ### Calculate loss, weight and sum 
            output_train = model(x_train_local, x_train_global, x_train_local_cen, x_train_global_cen, x_train_star)
            weight_ = batch_weighting[y_train.data.view(-1).long()].view_as(y_train)
            loss = criterion(output_train, y_train)
            loss_class_weighted = loss * weight_
            loss_class_weighted = loss_class_weighted.mean()
            train_loss += loss_class_weighted

            ### Train model (zero gradients and back propogate results)
            optimiser.zero_grad()
            loss_class_weighted.backward()
            optimiser.step()

        ### Record training loss for this epoch (divided by size of training dataset)
        epoch_train_loss.append(train_loss.cpu().detach().numpy() / len(data_loader.dataset))

        ########################################################################
        ### And now use the validation set for this epoch
        ########################################################################

        ### Loop over batches
        val_pred, val_gt, val_loss, num_corr = [], [], 0, 0
        for x_val_data, y_val in val_loader:

            toi_amount = y_val.sum().item()
            batch_weighting = torch.tensor([toi_amount, (args.n_batches - toi_amount)])/args.n_batches
            
            ### Get local view, global view, and label for validation
            x_val_local, x_val_global, x_val_local_cen, x_val_global_cen, x_val_star = x_val_data
            x_val_local = Variable(x_val_local).type(torch.FloatTensor)
            x_val_global = Variable(x_val_global).type(torch.FloatTensor)
            x_val_local_cen = Variable(x_val_local_cen).type(torch.FloatTensor)
            x_val_global_cen = Variable(x_val_global_cen).type(torch.FloatTensor)
            x_val_star = Variable(x_val_star).type(torch.FloatTensor)

            ### Fix dimensions for next steps
            y_val = Variable(y_val).type(torch.FloatTensor)
            x_val_local = x_val_local.unsqueeze(1)
            x_val_global = x_val_global.unsqueeze(1)
            x_val_local_cen = x_val_local_cen.unsqueeze(1)
            x_val_global_cen = x_val_global_cen.unsqueeze(1)
            x_val_star = x_val_star.unsqueeze(1)
            y_val = y_val.unsqueeze(1)

            ### Calculate loss, weight and sum
            output_val = model(x_val_local, x_val_global, x_val_local_cen, x_val_global_cen, x_val_star)
            weight_ = batch_weighting[y_val.data.view(-1).long()].view_as(y_val)
            loss_val = criterion(output_val, y_val)
            loss_class_weighted_val = loss_val * weight_
            loss_class_weighted_val = loss_class_weighted_val.mean()
            val_loss += loss_class_weighted_val.data

            ### Get number of correct predictions using threshold of 0.5
            output_pred = output_val >= 0.5
            num_corr += output_pred.eq(y_val.byte()).sum().item()

            ### Record predictions of model and ground truth
            val_pred.append(output_val.data.cpu().numpy())
            val_gt.append(y_val.data.cpu().numpy())

        ### Record validation loss calculate for this epoch (divided by size of validation dataset)
        epoch_val_loss.append(val_loss.cpu().detach().numpy() / len(val_loader.dataset))

        ### Record validation accuracy (# correct predictions in val set) for this epoch
        epoch_val_acc.append(num_corr / len(val_loader.dataset))

        ### Calculate average precision for this epoch
        epoch_val_ap.append(average_precision_score(np.concatenate(val_gt).ravel(), np.concatenate(val_pred).ravel(), average=None))
        
    ### Concatenate final predictions and ground truths for validation set
    final_val_pred = np.concatenate(val_pred).ravel()
    final_val_gt = np.concatenate(val_gt).ravel()

    return epoch_train_loss, epoch_val_loss, epoch_val_acc, epoch_val_ap, final_val_pred, final_val_gt


################################################################################
# BEGIN PROGRAM
################################################################################


print("Training Model...")

### Define model
model = LCModel()
switch(args.model):
    case "all":
        model = AllModel()
        break
    case "cent":
        model = CentroidModel()
        break
    case "sdp":
        model = ParameterModel()
        break
    case "lc":
    default:
        # Ignore

### Learning rate
lr = args.r_learn

### Specify optimiser for learning to use for training
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

### Specify loss function to use for training
criterion = nn.BCELoss(reduce=False)

### Specify the batch size to use for training
batch_size = args.n_batches

### Number of epochs to use for training
n_epochs = args.n_epochs

### Fetch data using data loader
training_data = CustomDataLoader(args.input_dir, True)
validation_data = CustomDataLoader(args.input_dir, False)
data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

### Train model
loss_train_epoch, loss_val_epoch, acc_val_epoch, ap_val_epoch, pred_val_final, gt_val_final = train_model(n_epochs, data_loader, val_loader, model, criterion, optimiser)

########################################
####### CALCULATE STATISTICS ###########
########################################

### Setup screen output
print("\nCALCULATING METRICS...\n")

### Store data
print("GT VAL FIN: ")
print(gt_val_final)
print("PRED VAL FIN: ")
print(pred_val_final)
print("LOSS TRAIN:")
print(loss_train_epoch)
print("LOSS VAL:")
print(loss_val_epoch)
with open(args.output_dir+"lt_epoch.npy", "wb") as f:
    np.save(f, loss_train_epoch)
with open(args.output_dir+"lv_epoch.npy", "wb") as f:
    np.save(f, loss_val_epoch)
with open(args.output_dir+"acc_val.npy", "wb") as f:
    np.save(f, acc_val_epoch)
with open(args.output_dir+"gt_val.npy", "wb") as f:
    np.save(f, gt_val_final)
with open(args.output_dir+"pred_val.npy", "wb") as f:
    np.save(f, pred_val_final)
print("   average precision = {0:0.4f}\n".format(AP))

################################################################################
# Most statistics calculation is done in post with the above saved files
################################################################################
 
### Calculate average precision and precision-recall curve
AP = average_precision_score(gt_val_final, pred_val_final, average=None)
print("Average precision: {0:0.2f}".format(AP))
P, R, _ = precision_recall_curve(gt_val_final, pred_val_final)

### calculate confusion matrix based on different thresholds 
thresh = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.975, 0.999]
prec_thresh, recall_thresh = np.zeros(len(thresh)), np.zeros(len(thresh))
for n, nval in enumerate(thresh):
    pred_byte = np.zeros(len(pred_val_final))
    for i, val in enumerate(pred_val_final):
        if val > nval:
            pred_byte[i] = 1.0
        else:
            pred_byte[i] = 0.0
    prec_thresh[n] = precision_score(gt_val_final, pred_byte)
    recall_thresh[n] = recall_score(gt_val_final, pred_byte)
    print("   thresh = {0:0.2f}, precision = {1:0.2f}, recall = {2:0.2f}".format(thresh[n], prec_thresh[n], recall_thresh[n]))
    tn, fp, fn, tp = confusion_matrix(gt_val_final, pred_byte).ravel()
    print("      TN = {0:0}, FP = {1:0}, FN = {2:0}, TP = {3:0}".format(tn, fp, fn, tp))

model_path = args.output_dir + "test_model.pth"
torch.save(model.state_dict(), model_path)
print("\nOUTPUTTING MODEL + RESULTS @ " + model_path + "\n")
print("Finished")
