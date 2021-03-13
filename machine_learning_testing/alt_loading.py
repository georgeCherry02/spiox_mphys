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

### plotting packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


################################################################################
# PARSE ARGUMENTS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("n_epochs", help="number of epochs to use for training", type=int)
parser.add_argument("n_batches", help="number of batches to use for training", type=int)
parser.add_argument("r_learn", help="learning rate for Adam optimiser", type=float)
parser.add_argument("input_dir", help="Directory containing data required")
parser.add_argument("output_dir", help="Directory for data output")
args = parser.parse_args()

################################################################################
# DEFINE CLASSES
################################################################################

class CustomDataLoader(Dataset):

    def __init__(self, filepath):

        ### list of global, local, and info files (assumes certain names of files)
        self.file_list_local = np.sort(glob.glob(os.path.join(filepath, 'processed_lcs_and_centroids/*-local.npy')))
        self.file_list_global = np.sort(glob.glob(os.path.join(filepath, 'processed_lcs_and_centroids/*-global.npy')))
        self.info_file = os.path.join(filepath, 'collated_scientific_domain_parameters.csv')
        self.tce_data = pd.read_csv(self.info_file)

        ### gather up tce ids
        self.ids = self.tce_data["tce_id"]

        print("Len ids: "+str(len(self.ids)))
        print("Len files: "+str(len(self.file_list_local))+"/"+str(len(self.file_list_global)))

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### grab local and global views
        tce_id = self.ids[idx]
        data_local = np.load(self.file_list_local[idx])
        data_global = np.load(self.file_list_global[idx])

        ### get info file
        data_info = pd.read_csv(self.info_file)
        print(tce_id)
        tce_data_info = data_info[data_info["tce_id"]==tce_id]
        print(tce_data_info)
        temp = tce_data_info["pc"]
        pc_class = tce_data_info["pc"].values[0]

        # Remove classification and information columns
        del tce_data_info["pc"]
        del tce_data_info["tic_id"]
        del tce_data_info["tce_id"]

        # Remove calculated values
        del tce_data_info["ratio_of_mes_to_expected_mes"]
        del tce_data_info["log_ratio_of_planet_to_earth_radius"]
        del tce_data_info["log_duration_over_expected_duration"]

        # Remove TCE data
        del tce_data_info["semi_major_scaled_to_stellar_radius"]
        del tce_data_info["number_of_transits"]
        del tce_data_info["ingress_duration"]
        del tce_data_info["impact_parameter"]
        del tce_data_info["ratio_of_planet_to_star_radius"]

        # Remove Header data
        del tce_data_info["band_magnitude"]
        del tce_data_info["stellar_radius"]
        del tce_data_info["total_proper_motion"]
        del tce_data_info["stellar_log_g"]
        del tce_data_info["stellar_melaticity"]
        del tce_data_info["effective_temperature"]

        np_tce_data = tce_data_info.to_numpy()
        np_tce_data.astype(np.float)
        return(data_local[0], data_global[0], data_local[1], data_global[1], np_tce_data[0]), pc_class

class TestModel(nn.Module):
    
    def __init__(self):

        ### initialise model
        super(TestModel, self).__init__()

        ### define global convolutional layer
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
        
        ### define local convolutional layer
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

        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(16576, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            ### need output of 1 because using BCE for loss
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_local_cen, x_global_cen):
            
        ### concatonate light curve and centroid data
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)

        ### get outputs of global and local convolutional layers
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)

        ### flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### concatonate global and local views with stellar parameters
        out = torch.cat([out_global, out_local], dim=1)
        out = self.final_layer(out)

        return out

class TestModelSmall(nn.Module):

    '''
    
    PURPOSE: DEFINE EXTRANET-XS MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES AND CENTROID CURVES, STELLAR PARAMETERS
    OUTPUT: BINARY CLASSIFIER
    
    '''

    def __init__(self):

        ### initializing the nn.Moduel (super) class
        ### (must do this first always)
        super(TestModelSmall, self).__init__()

        ### define global convolutional lalyer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 8, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 16, 5, stride=1, padding=2),
            nn.ReLU(),
#            nn.MaxPool1d(2, stride=2),
#            nn.Conv1d(8, 16, 5, stride=1, padding=2),
#            nn.ReLU(),
        )

        ### define the local convolutional layer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 8, 5, stride=1, padding=2),
            nn.ReLU(),
        )
                
        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(24, 1),
            nn.Sigmoid())
        
    ### define how to move forward through model
    def forward(self, x_local, x_global, x_local_cen, x_global_cen, stellar_param):

        ### concatonate light curve and centroid data
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)
        
        ### do global pooling
        out_global = nn.functional.max_pool1d(out_global, out_global.shape[-1])
        out_local = nn.functional.max_pool1d(out_local, out_local.shape[-1])

        ### flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### concatonate global and local views with stellar parameters
        #out = torch.cat([out_global, out_local, stellar_param.squeeze(1)], dim=1)
        out = torch.cat([out_global, out_local], dim=1)
        out = self.final_layer(out)

        return out


class TestModelSmallNoMtm(nn.Module):

    '''
    
    PURPOSE: DEFINE EXTRANET-XS MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES AND CENTROID CURVES, STELLAR PARAMETERS
    OUTPUT: BINARY CLASSIFIER
    
    '''

    def __init__(self):

        ### initializing the nn.Moduel (super) class
        ### (must do this first always)
        super(TestModelSmallNoMtm, self).__init__()

        ### define global convolutional lalyer
        self.fc_global = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=1, padding=2),
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
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )

        ### define local convolutional lalyer
        self.fc_local = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=1, padding=2),
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

        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(288, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            ### need output of 1 because using BCE for loss
            nn.Linear(512, 1),
            nn.Sigmoid())
        
    ### define how to move forward through model
    def forward(self, x_local, x_global):

        ######################################
        # These are the lines changed
        ######################################
        out_global = self.fc_global(x_global)
        out_local = self.fc_local(x_local)
        
        ### do global pooling
        out_global = nn.functional.max_pool1d(out_global, out_global.shape[-1])
        out_local = nn.functional.max_pool1d(out_local, out_local.shape[-1])

        ### flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### concatonate global and local views with stellar parameters
        out = torch.cat([out_global, out_local], dim=1)
        out = self.final_layer(out)

        return out

################################################################################
# Define auxillary functions
################################################################################

def train_model(n_epochs, data_loader, val_loader, model, criterion, optimiser):

    ### empty arrays to fill per-epoch outputs
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_val_ap = []

    ### loop over number of epochs of training
    for epoch in tqdm(range(n_epochs)):

        ### loop over batches
        train_loss = torch.zeros(1)
        for x_train_data, y_train in data_loader:

            ### get local view, global view and label for training
            x_train_local, x_train_global, x_train_local_cen, x_train_global_cen, x_train_star = x_train_data
            x_train_local = Variable(x_train_local).type(torch.FloatTensor)
            x_train_global = Variable(x_train_global).type(torch.FloatTensor)
            x_train_local_cen = Variable(x_train_local_cen).type(torch.FloatTensor)
            x_train_global_cen = Variable(x_train_global_cen).type(torch.FloatTensor)
            x_train_star = Variable(x_train_star).type(torch.FloatTensor)
            y_train = Variable(y_train).type(torch.FloatTensor)

            ### fix dimnensions for next steps
            x_train_local = x_train_local.unsqueeze(1)
            x_train_global = x_train_global.unsqueeze(1)
            x_train_local_cen = x_train_local_cen.unsqueeze(1)
            x_train_global_cen = x_train_global_cen.unsqueeze(1)
            x_train_star = x_train_star.unsqueeze(1)
            y_train = y_train.unsqueeze(1)

            ### calculate loss using model
            # Filtered out x_train_local_cen, global_cen and x_train_star
            output_train = model(x_train_local, x_train_global)
            loss = criterion(output_train, y_train)
            train_loss += loss.data

            ### train model (zero gradients and back propogate results)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        ### record training loss for this epoch (divided by size of training dataset)
        epoch_train_loss.append(train_loss.cpu().numpy() / len(data_loader.dataset))

        ### for validation set

        ### loop over batches
        val_pred, val_gt, val_loss, num_corr = [], [], 0, 0
        for x_val_data, y_val in val_loader:
            
            ### get local view, global view, and label for validation
            x_val_local, x_val_global, x_val_local_cen, x_val_global_cen, x_val_star = x_val_data
            x_val_local = Variable(x_val_local).type(torch.FloatTensor)
            x_val_global = Variable(x_val_global).type(torch.FloatTensor)
            x_val_local_cen = Variable(x_val_local_cen).type(torch.FloatTensor)
            x_val_global_cen = Variable(x_val_global_cen).type(torch.FloatTensor)
            x_val_star = Variable(x_val_star).type(torch.FloatTensor)

            ### fix dimensions for next steps
            y_val = Variable(y_val).type(torch.FloatTensor)
            x_val_local = x_val_local.unsqueeze(1)
            x_val_global = x_val_global.unsqueeze(1)
            x_val_local_cen = x_val_local_cen.unsqueeze(1)
            x_val_global_cen = x_val_global_cen.unsqueeze(1)
            x_val_star = x_val_star.unsqueeze(1)
            y_val = y_val.unsqueeze(1)

            ### calculate loss & add to sum over all batches
            output_val = model(x_val_local, x_val_global)
            loss_val = criterion(output_val, y_val)
            val_loss += loss_val.data

            ### get number of correct predictions using threshold of 0.5
            output_pred = output_val >= 0.5
            num_corr += output_pred.eq(y_val.byte()).sum().item()

            ### record predictions and ground truth bty model
            val_pred.append(output_val.data.cpu().numpy())
            val_gt.append(y_val.data.cpu().numpy())

        ### record validation loss calculate for this epoch (divided by size of validation dataset)
        epoch_val_loss.append(val_loss.cpu().numpy() / len(val_loader.dataset))

        ### record validation accuracy (# correct predictions in val set) for this epoch
        epoch_val_acc.append(num_corr / len(val_loader.dataset))

        ### calculate average precision for this epoch
        epoch_val_ap.append(average_precision_score(np.concatenate(val_gt).ravel(), np.concatenate(val_pred).ravel(), average=None))
        
    ### grab final predictions and ground truths for validation set
    final_val_pred = np.concatenate(val_pred).ravel()
    final_val_gt = np.concatenate(val_gt).ravel()

    return epoch_train_loss, epoch_val_loss, epoch_val_acc, epoch_val_ap, final_val_pred, final_val_gt


################################################################################
# BEGIN PROGRAM
################################################################################


print("Training Model...")

### define model
model = TestModelSmallNoMtm()

### learning rate
lr = args.r_learn

### specify optimiser for learning to use for training
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

### specify loss function to use for training
criterion = nn.BCELoss(reduce=False)

### specify the batch size to use for training
batch_size = args.n_batches

### number of epochs to use for training
n_epochs = args.n_epochs

### grab data using data loader
### yes these are the same, testing to see that it will identify one thing it's trained by
training_data = CustomDataLoader(args.input_dir)
validation_data = CustomDataLoader(args.input_dir)
data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

### train model
loss_train_epoch, loss_val_epoch, acc_val_epoch, ap_val_epoch, pred_val_final, gt_val_final = train_model(n_epochs, data_loader, val_loader, model, criterion, optimiser)
"""
print(loss_train_epoch)
print(loss_val_epoch)
print(acc_val_epoch)
print(ap_val_epoch)
print(pred_val_final)
print(gt_val_final)
"""

################################################################################
# ALL EXONET CODE
################################################################################

########################################
####### CALCULATE STATISTICS ###########
########################################

### setup screen output
print("\nCALCULATING METRICS...\n")

### calculate average precision & precision-recall curves
AP = average_precision_score(gt_val_final, pred_val_final, average=None)
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
 
### calculate precision-recall curve
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
    # To avoid breaking???
    # a = confusion_matrix(gt_val_final, pred_byte).ravel()
    # print("Confusion matrix:")
    # print(a)

"""
########################################
######### OUTPUT MODEL + STATS  ########
########################################

### transform from loss per sample to loss per batch (multiple by batch size to compare to Chris')
loss_train_batch = [x.item()* batch_size for x in loss_train_epoch]
loss_val_batch = [x.item()* batch_size for x in loss_val_epoch]

### setup output
run = 0

### output predictions & ground truth
pt_fname = os.path.join(args.m_out, 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-lr' + str(lr) + '-pt.csv')
while os.path.isfile(pt_fname):
    run +=1
    pt_fname = os.path.join(args.m_out, 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-lr' + str(lr) + '-pt.csv')
df = pd.DataFrame({"gt" : gt_val_final, "pred" : pred_val_final})
df.to_csv(pt_fname, index=False)

### output per-iteration values
epochs_fname = os.path.join(args.m_out, 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-lr' + str(lr) + '-epoch.csv')
df = pd.DataFrame({"loss_train":loss_train_batch, "loss_val":loss_val_batch, "acc_val":acc_val_epoch, "ap_val":ap_val_epoch})
df.to_csv(epochs_fname, index=False)

### save model
model_fname = os.path.join(args.m_out, 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-lr' + str(lr) + '-model.pth')
torch.save(model.state_dict(), os.path.join(args.m_out, model_fname))
print("\nOUTPUTTING MODEL + RESULTS @ " + os.path.join(args.m_out, model_fname) + "\n")


########################################
################ MAKE PLOTS ############
########################################

### setup figure
fig = plt.figure(figsize=(7, 7))
ax = gridspec.GridSpec(2,2)
ax.update(wspace = 0.4, hspace = 0.4)
ax1 = plt.subplot(ax[0,0])
ax2 = plt.subplot(ax[0,1])
ax3 = plt.subplot(ax[1,0])
ax4 = plt.subplot(ax[1,1])

### plot precision-recall curve
ax1.set_xlabel('Precision', fontsize=10, labelpad=10)
ax1.set_ylabel('Recall', fontsize=10)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.plot(R, P, linewidth=3, color='black')

### plot loss curve for training and validation sets
ax2.set_xlabel('Epoch', fontsize=10, labelpad=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.set_xlim([0.0, n_epochs])
ax2.set_ylim([0.0, np.max(loss_train_batch)*1.5])
ax2.plot(np.arange(len(loss_train_batch)), loss_train_batch, linewidth=3, color='cadetblue')
ax2.plot(np.arange(len(loss_val_batch)), loss_val_batch, linewidth=3, color='orangered')

### plot average precision per epoch
ax3.set_xlabel('Epoch', fontsize=10, labelpad=10)
ax3.set_ylabel('Average Precision', fontsize=10)
ax3.plot(np.arange(len(ap_val_epoch)), ap_val_epoch, linewidth=1.0, color='orangered')
ax3.scatter(np.arange(len(ap_val_epoch)), ap_val_epoch, marker='o', edgecolor='orangered', facecolor='orangered', s=10, linewidth=0.5, alpha=0.5)

### plot accuracy per epoch
ax4.set_xlabel('Epoch', fontsize=10, labelpad=10)
ax4.set_ylabel('Accuracy', fontsize=10)
ax4.plot(np.arange(len(acc_val_epoch)), acc_val_epoch, color='orangered', linewidth=1.0)
ax4.scatter(np.arange(len(acc_val_epoch)), acc_val_epoch, marker='o', edgecolor='orangered', facecolor='orangered', s=10, linewidth=0.5, alpha=0.5)

### save plot
plot_fname = 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-plot.pdf'
plt.savefig(os.path.join(args.m_out, plot_fname), bbox_inches='tight', dpi=200, rastersized=True, alpha=True)

"""
print("Finished")
