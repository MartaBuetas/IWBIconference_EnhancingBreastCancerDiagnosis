### BINARY EXPERIMENT
## Classes: Healthy / Suspicious (binary problem)
## Balancing: only for the train set #healthy=#non healthy with Weighted Random Sampler
## 3 folds computed

task='Binary'

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, train_test_split
import seaborn as sns
from datasets import load_dataset
from PIL import Image
import copy
from torch.optim.lr_scheduler import StepLR
import copy
import time
from torch.nn.modules.loss import BCEWithLogitsLoss 
import itertools
from torch import optim
from tqdm import tqdm
from torchvision.models import resnet50,  ResNet50_Weights
import random
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

"""# Load data"""
root=input("Path of patch dataset folder: ")

##
dataset = load_dataset("imagefolder", data_dir=root)

"""Reading metadata"""

# Load the metadata from the JSONL file

metadata_path=os.path.join(root, 'metadata.jsonl')
with open(metadata_path, "r") as f:
    metadata = [json.loads(line) for line in f]

# Convert the list of dictionaries into a pandas dataframe
metadata_df = pd.DataFrame([sample for sample in metadata])


"""Remove those metadata entries without image:"""

# Iterate over the list of dictionaries and check if the image file exists
new_data = []

for sample in metadata:
    image_file = sample['file_name']
    image_path=os.path.join(root, image_file)
    if os.path.exists(image_path):
        new_data.append(sample)

metadata=new_data

md = metadata
data = copy.deepcopy(dataset)

# Extract the labels and filenames from the metadata
healthy = 0
benign = 0
malign = 0
filenames = []
format =[]
patients=[]
for sample in md:
    if 'suspicious' in sample and 'classification' in sample and 'file_name' in sample and 'Format' in sample:
      if sample['suspicious']== 0:
        healthy+=1
      if sample['classification']== ' Benign ':
        benign+=1
      if sample['classification']== ' Malign ':
        malign+=1
      filenames.append(sample['file_name'])
      format.append(sample['Format'])
      patients.append(sample['patient_id'])

print('# of unique patients: ',len(set(patients)))

"""As the patient_ids have to be integers in order to be able to handle them as a group, I will map each patient_id to an unique integer."""

# Define a function to update 'suspicious' to 'healthy'
def update_suspicious_to_healthy(example):
    example['healthy'] = 1 - example['suspicious']
    example.pop('suspicious')
    return example

# Apply the function to the 'train' split of the dataset
data['train'] = data['train'].map(update_suspicious_to_healthy)

data_df = data['train'].to_pandas()
# Reset the index of the DataFrame to consecutive integers
data_df = data_df.reset_index(drop=True)

# Compute the integer patient IDs using pd.factorize
int_patient_ids = pd.factorize(data_df['patient_id'])[0]
data=data.map(lambda example, idx: {"int_patient_id": int_patient_ids[idx]}, with_indices=True)

md_df = pd.DataFrame(md)

# Create a mapping of unique patient_ids to integers
id_map = {patient_id: i for i, patient_id in enumerate(md_df['patient_id'].unique())}

# Map patient_ids to integer values and create a new column
md_df['int_patient_id'] = md_df['patient_id'].map(id_map)
md = md_df.to_dict(orient='records')

'''CLASSIFIER SETUP'''

#Load ResNet50 trained weights on ImageNet 
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Some needed transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(lambda x: preprocess(x))
])

# Define a custom dataset class for the sets
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.images = dataset['image']
        self.healthy = dataset['healthy']
        self.Format = dataset['Format']
        self.zoom_group = dataset['zoom_group']
        self.density = dataset['density']
        self.classification = dataset['classification']
        self.image_view = dataset['image_view']
        self.nodule = dataset['nodule']
        self.calcification = dataset['calcification']
        #self.shape = dataset.shape
        #self.meta = [metadata[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.convert('RGB')
        image = transform(image)

        label = self.healthy[idx]
        metadata = {
            'format': self.Format[idx],
            'zoom_group': self.zoom_group[idx],
            'density': self.density[idx],
            'classification': self.classification[idx],
            'image_view': self.image_view[idx],
            'calcification': self.calcification[idx],
            'nodule':self.nodule[idx]
        }
        
        return image, label, metadata

def create_weightsamplers(train_custom_set, val_custom_set):
  if task=='binary':
    num_majority_train = sum(train_custom_set.labels) # sum of the '1's
    num_minority_train = len(train_custom_set) - num_majority_train

    undersample_train = torch.utils.data.WeightedRandomSampler(
        [1 if healthy == 0 else num_minority_train / num_majority_train for healthy in train_custom_set.healthy],
        num_samples=len(train_custom_set),
        replacement=True
    )
  else:
    '''Train undersamples'''
    malign_mask = [label == ' Malign ' for label in train_custom_set.classification]
    benign_mask = [label == ' Benign ' for label in train_custom_set.classification]
    normal_mask = [label == 'Normal' for label in train_custom_set.classification]

    # number of occurrences of each class label
    num_malign_train = sum(malign_mask)
    num_benign_train = sum(benign_mask)
    num_normal_train = sum(normal_mask)

    # total number of samples in the dataset
    total_samples = len(train_custom_set)

    weight_malign = 1 / num_malign_train
    weight_benign = 1 / num_benign_train
    weight_normal = 1 / num_normal_train

    # normalize the weights so that they sum up to 1
    total_weight = weight_malign + weight_benign + weight_normal
    weight_malign /= total_weight
    weight_benign /= total_weight
    weight_normal /= total_weight

    # ist of weights corresponding to each sample in the dataset
    weights_train = [weight_malign if malign else weight_benign if benign else weight_normal for malign, benign in zip(malign_mask, benign_mask)]

    # sampler that samples from the dataset with replacement
    undersampler_train = WeightedRandomSampler(weights=weights_train, num_samples=len(train_custom_set), replacement=True)

    '''Validation undersamples'''
    malign_mask = [label == ' Malign ' for label in val_custom_set.classification]
    benign_mask = [label == ' Benign ' for label in val_custom_set.classification]
    normal_mask = [label == 'Normal' for label in val_custom_set.classification]

    # number of occurrences of each class label
    num_malign_val = sum(malign_mask)
    num_benign_val= sum(benign_mask)
    num_normal_val = sum(normal_mask)

    # total number of samples in the dataset
    total_samples = len(val_custom_set)

    weight_malign = 1 / num_malign_val
    weight_benign = 1 / num_benign_val
    weight_normal = 1 / num_normal_val

    # Normalize the weights so that they sum up to 1
    total_weight = weight_malign + weight_benign + weight_normal
    weight_malign /= total_weight
    weight_benign /= total_weight
    weight_normal /= total_weight

    # list of weights corresponding to each sample in the dataset
    weights_val = [weight_malign if malign else weight_benign if benign else weight_normal for malign, benign in zip(malign_mask, benign_mask)]

    # sampler that samples from the dataset with replacement
    undersampler_val = WeightedRandomSampler(weights=weights_val, num_samples=len(val_custom_set), replacement=True)
  return undersampler_train, undersampler_val


def misclassification_rates(misclassified_metadata, df, display_hist=False):
  '''Misclassification rates = #of misclassified samples for class X / # of total samples in the test set of class X '''
  '''Count of samples for each class in the training set in order to calculate misclassification rates'''
  nodule_count=(df['nodule']==1).sum()
  nodule_num_classes=len(set(df['nodule'].to_list()))

  calcification_count=(df['calcification']==1).sum()
  calcification_num_classes=len(set(df['calcification'].to_list()))

  density_1_count=(df['density']==1).sum()
  density_2_count=(df['density']==2).sum()
  density_3_count=(df['density']==3).sum()
  density_4_count=(df['density']==4).sum()
  density_count = [density_1_count, density_2_count, density_3_count, density_4_count]
  density_num_classes=len(set(df['density'].to_list()))

  zoom_group_1_count=(df['zoom_group']==1).sum()
  zoom_group_2_count=(df['zoom_group']==2).sum()
  zoom_group_3_count=(df['zoom_group']==3).sum()
  zoom_group_count = [zoom_group_1_count, zoom_group_2_count, zoom_group_3_count]
  zoom_group_num_classes=len(set(df['zoom_group'].to_list()))

  image_view_0_count=(df['image_view']==0).sum()
  image_view_1_count=(df['image_view']==1).sum()
  image_view_2_count=(df['image_view']==2).sum()
  image_view_3_count=(df['image_view']==3).sum()
  image_view_4_count=(df['image_view']==4).sum()
  image_view_count = [image_view_0_count, image_view_1_count, image_view_2_count, image_view_3_count, image_view_4_count]
  image_view_num_classes=len(set(df['image_view'].to_list()))

  format_digital_count=(df['Format']=='Digital').sum()
  format_num_classes=len(set(df['Format'].to_list()))

  '''Histograms of misclassified samples metadata: '''
  density = misclassified_metadata['density']
  zoom_group = misclassified_metadata['zoom_group']
  image_view = misclassified_metadata['image_view']
  format = misclassified_metadata['format']
  calcification = misclassified_metadata['calcification']
  nodule_features = misclassified_metadata['nodule']

  fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

  hist, _, _ = axs[0, 0].hist(density, bins=density_num_classes, color='blue')
  axs[0, 0].set_title('Density')
  density_misclassification_rate=np.zeros(density_num_classes)
  for i in (range(density_num_classes)):
      # calculate misclassification rate for this bin and class
      density_misclassification_rate[i] = int(hist[i])/int(density_count[i])

  hist, _, _ = axs[0, 1].hist(zoom_group, bins=zoom_group_num_classes, color='green')
  axs[0, 1].set_title('Zoom group')
  zoom_group_misclassification_rate = np.zeros(zoom_group_num_classes)
  for i in (range(zoom_group_num_classes)):
      if int(zoom_group_count[i]) != 0:
        # calculate misclassification rate for this bin and class
        zoom_group_misclassification_rate[i] = int(hist[i])/int(zoom_group_count[i])

  # image_view histogram
  hist, _, _ = axs[0, 2].hist(image_view, bins=image_view_num_classes, color='red')
  axs[0, 2].set_title('Image view')
  image_view_misclassification_rate=np.zeros(image_view_num_classes)
  for i in (range(image_view_num_classes)):
      # calculate misclassification rate for this bin and class
      image_view_misclassification_rate[i] = int(hist[i])/int(image_view_count[i])
  
  # format histogram
  hist, _, _ = axs[1, 0].hist(format, bins=format_num_classes, color='purple')
  axs[1, 0].set_title('Format')
  # calculate misclassification rate for FILM
  film_misclassification_rate = int(hist[0])/int(len(train_df)-format_digital_count)
  # calculate misclassification rate for DIGITAL
  digital_misclassification_rate = int(hist[1])/int(format_digital_count)

  # calcification histogram
  hist, _, _ = axs[1, 1].hist(calcification, bins=calcification_num_classes, color='orange')
  axs[1, 1].set_title('Calcification')
  # calculate misclassification rate for NO CALCIFICATION
  no_calcification_misclassification_rate = int(hist[0])/int(len(train_df)-calcification_count)
  # calculate misclassification rate for CALCIFICATION
  calcification_misclassification_rate = int(hist[1])/int(calcification_count)

  # nodule features histogram
  hist, _, _ = axs[1, 2].hist(nodule_features, bins=nodule_num_classes, color='gray')
  axs[1, 2].set_title('Nodule')
  # calculate misclassification rate for NO NODULE
  no_nodule_misclassification_rate = int(hist[0])/int(len(train_df)-nodule_count)
  # calculate misclassification rate for CALCIFICATION
  nodule_misclassification_rate = int(hist[1])/int(nodule_count)

  if display_hist:
    fig.suptitle('Misclassified samples distribution - Validation', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig('misclassified_metadata_distribution.png')
    plt.clf()

  return density_misclassification_rate, zoom_group_misclassification_rate, image_view_misclassification_rate, film_misclassification_rate, digital_misclassification_rate, calcification_misclassification_rate, no_calcification_misclassification_rate, nodule_misclassification_rate, no_nodule_misclassification_rate

def make_train_step(model, optimizer, criterion):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = criterion(yhat.squeeze(),y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return yhat, loss
  return train_step



def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    train_losses = []
    val_losses = []

    val_accuracies = []
    #val_precisions = []
    #val_recalls = []

    train_accuracies = []
    #train_precisions = []
    #train_recalls = []

    early_stopping_tolerance = 100
    early_stopping_threshold = 0.001
    early_stopping=False

    best_loss = float('inf')
    best_model_wts = None

    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print('\nRunning epoch : {}'.format(epoch + 1)) 
        epoch_loss = 0
        epoch_correct_predictions = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):  # iterate over batches
            x_batch, y_batch, m_batch = data

            y_batch = y_batch.float()
            y_hat, loss = train_step(x_batch, y_batch)
            
            epoch_loss += loss / len(train_loader)

            predicted_probs = torch.sigmoid(y_hat)
            # Round predicted probabilities to obtain binary predictions (0 or 1)
            predicted_binary = torch.round(predicted_probs)
            predicted = predicted_binary.detach()

            predicted_labels_binary = torch.round(predicted)

            # Create the mask for correct and wrong predictions
            correct_mask = torch.eq(predicted_labels_binary, y_batch)

            epoch_correct_predictions += torch.sum(correct_mask)      

        train_losses.append(epoch_loss.item())

        train_acc = epoch_correct_predictions / len(train_loader.dataset)  # calculate train accuracy for the epoch
        train_accuracies.append(train_acc)
        
        y_true_flat = np.array(y_batch)
        y_pred_flat = np.array(predicted_labels_binary)

        if epoch == num_epochs-1:
          train_precision = precision_score(y_true_flat, y_pred_flat, average=None, zero_division=0.0)
          train_recall = recall_score(y_true_flat, y_pred_flat, average=None)
        
          train_precision = np.array(train_precision)
          train_recall = np.array(train_recall)

          # Calculate mean of train_precisions and train_recalls
        
          print('\nTrain precision: {}\nTrain recall : {}'.format(train_precision, train_recall))
          # Compute ROC AUC using y_pred as predicted class labels      
          train_roc_auc = []
          y_true=y_batch.numpy()
          y_pred=y_hat.detach().numpy()
          
          train_roc_auc = roc_auc_score(y_true, np.array(predicted_probs.detach()))

          print('Training ROC AUC: ',train_roc_auc)

        print('\nEpoch : {}, train loss: {:.4f}, train accuracy : {:.4f}'.format(epoch + 1, epoch_loss, train_acc))
        
        # validation doesn't require gradient
        with torch.no_grad():
            cum_loss = 0
            correct_predictions = 0
            total_predictions = 0
      
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):  # iterate over batches
       
                x_batch, y_batch, m_batch = data
                #print('\nm_batch' , m_batch)

                y_batch = y_batch.float()
                #y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape

                model.eval()

                y_pred = model(x_batch)
                val_loss = criterion(y_pred.squeeze(), y_batch)
                cum_loss += val_loss.item() / len(val_loader)

                predicted = torch.sigmoid(y_pred)
                # Round predicted probabilities to obtain binary predictions (0 or 1)
                predicted_binary = torch.round(predicted)
                predicted = predicted_binary.detach()
     
                predicted_labels_binary = torch.round(predicted)

                # Create the mask for correct and wrong predictions
                correct_mask = torch.eq(predicted_labels_binary, y_batch)
                
                '''Extract metadata info of misclassified samples'''
                if epoch == (num_epochs-1):  
                  # Get the metadata of the misclassified samples
                  # Convert correct_mask tensor to numpy array
                  correct_mask_np = correct_mask.numpy()

                  # Filter misclassified samples using correct_mask_np
                  misclassified_indices = np.where(correct_mask_np == False)[0]

                  # Extract metadata for misclassified samples from the current batch
                  misclassified_metadata_batch = {key: [val[i] for i in misclassified_indices] for key, val in m_batch.items()}

                  #misclassified_indices = correct_mask == False
                  #misclassified_metadata_batch = {key: [val[i] for i, misclass in enumerate(misclassified_indices) if misclass] for key, val in m_batch.items()}
                  if i==0:
                     misclassified_metadata=misclassified_metadata_batch
                  else:
                    for key in misclassified_metadata:
                      misclassified_metadata[key].extend(misclassified_metadata_batch[key])

                correct_predictions += torch.sum(correct_mask)   
                total_predictions += y_batch.size(0)

            accuracy = correct_predictions / total_predictions
            val_losses.append(cum_loss)
            val_accuracies.append(accuracy)
            print('\nValidation loss: {:.4f}, val accuracy : {:.4f}'.format(cum_loss, accuracy))

            if epoch == num_epochs-1:
              y_true_flat = np.array(y_batch)
              y_pred_flat = np.array(predicted_labels_binary)

              val_precision = precision_score(y_true_flat, y_pred_flat, average=None, zero_division=0.0)
              val_recall = recall_score(y_true_flat, y_pred_flat, average=None)
              
              #val_precisions.append(val_precision)
              #val_recalls.append(val_recall)

              val_precision = np.array(val_precision)
              val_recall = np.array(val_recall)

              # Calculate mean of train_precisions and train_recalls
              mean_val_precision = np.mean(val_precision)
              mean_val_recall = np.mean(val_recall)

              print('\nValidation precision: {}, mean: {:.4f} \nVal recall : {}, mean: {:.4f}'.format(val_precision, mean_val_precision, val_recall, mean_val_recall))

              val_roc_auc = roc_auc_score(y_true_flat, np.array(predicted.detach()))

              print('Validation ROC AUC: ', val_roc_auc)
              
            #save best model
            if cum_loss <= best_loss:
              best_loss = cum_loss
              best_model_wts = model.state_dict()
              early_stopping_counter = 0
            else:
              early_stopping_counter +=1
        
            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
              print("\nTerminating: early stopping")
              early_stopping=True
              epochs=epoch
              break #terminate training
           
    if early_stopping==False: 
       epochs=num_epochs
  
    #load best model
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_precision, val_precision, train_recall, val_recall, train_roc_auc, val_roc_auc, epochs, misclassified_metadata


'''Test function'''
def test(test_model):
    test_model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    y_score_probs = torch.tensor([])

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, m_batch = data
            outputs = test_model(inputs)

            targets = targets.float()
            outputs = torch.sigmoid(outputs)
            predicted_binary = torch.round(outputs)
            predicted = predicted_binary.detach()
            predicted_labels_binary = torch.round(predicted)
        
            y_true = torch.cat((y_true, targets), 0)
            y_score_probs = torch.cat((y_score_probs, outputs.detach()), 0)
            y_score = torch.cat((y_score, predicted_labels_binary), 0)
            
            correct_mask = torch.eq(predicted_labels_binary, targets)
            
            '''Extract metadata info of misclassified samples'''  
            correct_mask_np = correct_mask.numpy()

            # Filter misclassified samples using correct_mask_np
            misclassified_indices = np.where(correct_mask_np == False)[0]

            # Extract metadata for misclassified samples from the current batch
            misclassified_metadata_batch = {key: [val[i] for i in misclassified_indices] for key, val in m_batch.items()}

            if i==0:
                misclassified_metadata=misclassified_metadata_batch
            else:
              for key in misclassified_metadata:
                misclassified_metadata[key].extend(misclassified_metadata_batch[key])

        
        
    print('==> Evaluating model ...')
    #Print roc_auc socre
    test_roc_auc = roc_auc_score(y_true, np.array(y_score_probs))
    fpr, tpr, thresh = roc_curve(y_true, np.array(y_score_probs))
    print('\n Test roc_auc_scores for the three classes: ',test_roc_auc)
    
    # Convert to class labels
    y_true = np.array(y_true)
    y_pred = np.array(y_score)

    # Compute accuracy
    test_accuracy = accuracy_score(y_true, y_pred)

    # Compute precision, recall, and F1-score
    test_precision, test_recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    return test_roc_auc, fpr, tpr, thresh, test_accuracy, test_precision, test_recall, misclassified_metadata


# Some hyperparameters:
BATCH_SIZE=128
num_epochs = 100
learning_rate = 0.01
num_folds=3

"""Train, val split with the patients as groups and stratified with respect to labels and formats."""

healthy = data['train']['healthy']
groups = data['train']["int_patient_id"]
formats=data['train']["Format"]
zoom_groups = data['train']["zoom_group"]

healthy = np.asarray(healthy)
groups = np.asarray(groups)
formats = np.asarray(formats)
zoom_groups = np.asarray(zoom_groups)

# Perform filtering based on user input for train and validation sets
filter_zoom_1_train_val = input("Filter Zoom Group 1 for train and val sets? (y/n): ").lower() == 'y'
filter_zoom_2_train_val = input("Filter Zoom Group 2 for train and val sets? (y/n): ").lower() == 'y'
filter_zoom_3_train_val = input("Filter Zoom Group 3 for train and val sets? (y/n): ").lower() == 'y'

# Perform filtering based on user input for test set
filter_zoom_1_test = input("Filter Zoom Group 1 for test set? (y/n): ").lower() == 'y'
filter_zoom_2_test = input("Filter Zoom Group 2 for test set? (y/n): ").lower() == 'y'
filter_zoom_3_test = input("Filter Zoom Group 3 for test set? (y/n): ").lower() == 'y'

save_folder = input("Please enter the path of the folder where the results will be saved: ")

num_folds=input("How many folds do you want to compute? (enter an integer): ")
num_epochs=input("Number of epochs (enter an integer): ")
num_epochs=int(num_epochs)

# Perform a stratified train-validation-test split with grouping
splitter = GroupShuffleSplit(n_splits=int(num_folds), test_size=0.9, random_state=42)

best_val_roc_auc=0
cv_train_accuracies=[]
cv_val_accuracies=[]
cv_train_roc_auc=[]
cv_val_roc_auc=[]
cv_train_precisions=[]
cv_val_precisions=[]
cv_train_recalls=[]
cv_val_recalls=[]
cv_test_roc_aucs=[]
cv_test_precisions=[]
cv_test_accuracies=[]
cv_test_recalls=[]
cv_density_misclassification_rate=[]
cv_zoom_group_misclassification_rate=[]
cv_image_view_misclassification_rate=[]
cv_film_misclassification_rate=[]
cv_digital_misclassification_rate=[]
cv_calcification_misclassification_rate=[]
cv_no_calcification_misclassification_rate=[]
cv_nodule_misclassification_rate=[]
cv_no_nodule_misclassification_rate=[]

n_classes = 1

cv_fprs = []
cv_tprs = []
cv_thresh = []
i=0


for test_indices, train_val_indices in splitter.split(data['train'], healthy, groups=groups):
  #set new seed
  manualSeed = 42+i*5
  i=i+1
  print('\n\n==> Fold number : ', i, '/', num_folds)
  print("Random Seed: ", manualSeed)
  random.seed(manualSeed)
  torch.manual_seed(manualSeed)

  # Include samples with zoom_group '0' in the train_val_indices for all cases
  train_val_indices = train_val_indices[((zoom_groups[train_val_indices] == 1) & filter_zoom_1_train_val) |
                        ((zoom_groups[train_val_indices] == 2) & filter_zoom_2_train_val) |
                        ((zoom_groups[train_val_indices] == 3) & filter_zoom_3_train_val) |
                        (zoom_groups[train_val_indices] == 0)]


  test_indices = test_indices[((zoom_groups[test_indices] == 1) & filter_zoom_1_test) |
                        ((zoom_groups[test_indices] == 2) & filter_zoom_2_test) |
                        ((zoom_groups[test_indices] == 3) & filter_zoom_3_test) |
                        (zoom_groups[test_indices] == 0)]

  # Split the remaining indices into validation and test indices
  splitter_train = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

  train_val_dataset = data['train'].select(train_val_indices)

  train_indices, val_indices = next(splitter_train.split(train_val_dataset, healthy[train_val_indices], groups=groups[train_val_indices]))

  # Select examples from the filtered dataset using the train, validation, and test indices
  test_dataset = data['train'].select(test_indices)
  val_dataset = train_val_dataset.select(val_indices)
  train_dataset = train_val_dataset.select(train_indices)

  train_df = train_dataset.to_pandas()
  val_df = val_dataset.to_pandas()
  test_df = test_dataset.to_pandas()

  print("\n\nTRAIN HEALTHY RATIO:", healthy[train_indices].mean())
  print("\nVALIDATION HEALTHY RATIO:", healthy[val_indices].mean())
  print("\nTEST HEALTHY RATIO :", healthy[test_indices].mean())
  print("\n\nTRAIN    malign number of samples:", (train_df['classification'] == ' Malign ').sum())
  print("\nVALIDATION malign number of samples:", (val_df['classification'] == ' Malign ').sum())
  print("\nTEST       malign number of samples :", (test_df['classification'] == ' Malign ').sum())
  print("\n\nTRAIN    benign number of samples:", (train_df['classification'] == ' Benign ').sum())
  print("\nVALIDATION benign number of samples:", (val_df['classification'] == ' Benign ').sum())
  print("\nTEST       benign number of samples :", (test_df['classification'] == ' Benign ').sum())
  print("\n\nTRAIN    calcification number of samples:", (train_df['calcification'] == 1).sum())
  print("\nVALIDATION calcification number of samples:", (val_df['calcification'] == 1).sum())
  print("\nTEST       calcification number of samples :", (test_df['calcification'] == 1).sum())
  print("\n\nTRAIN    nodule number of samples:", (train_df['nodule'] == 1).sum())
  print("\nVALIDATION nodule number of samples:", (val_df['nodule'] == 1).sum())
  print("\nTEST       nodule number of samples :", (test_df['nodule'] == 1).sum())
  print("\n\nTRAIN GROUPS        :", len(set(train_dataset['int_patient_id'])))
  print("\nVALIDATION GROUPS  :", len(set(val_dataset['int_patient_id'])))
  print("\nTEST GROUPS         :", len(set(test_dataset['int_patient_id'])))

  print('\nThere are ',len(train_df['int_patient_id'].unique()), ' different patients in the train set.')
  print('\nThere are ',len(val_df['int_patient_id'].unique()), ' different patients in the validation set.')
  print('\nThere are ',len(test_df['int_patient_id'].unique()), ' different patients in the test set.')

  # Print number of examples in filtered train, validation, and test datasets
  print("\n\nNumber of examples in filtered train dataset:", len(train_dataset))
  print("\nNumber of examples in filtered validation dataset:", len(val_dataset))
  print("\nNumber of examples in filtered test dataset:", len(test_dataset))

  # Calculate the distribution of 'zoom_group' values in each filtered dataset
  train_zoom_group_counts = np.bincount(train_dataset['zoom_group'])
  val_zoom_group_counts = np.bincount(val_dataset['zoom_group'])
  test_zoom_group_counts = np.bincount(test_dataset['zoom_group'])

  print("\n\nZoom Group distribution in filtered train dataset:", train_zoom_group_counts)
  print("\nZoom Group distribution in filtered validation dataset:", val_zoom_group_counts)
  print("\nZoom Group distribution in filtered test dataset:", test_zoom_group_counts)

  '''Ratio of healthy/malign/normal in train dataset'''
  # Total healthy samples
  healthy_sum = healthy[train_indices].sum()
  # Total malign samples
  malign_sum=(train_df['classification'] == ' Malign ').sum()
  # Total benign samples
  benign_sum=(train_df['classification'] == ' Benign ').sum()

  # Healthy ratio
  healthy_ratio=healthy_sum/len(train_dataset)
  print('Healthy ratio: ', healthy_ratio)

  # Create a custom dataset object for the train dataset
  train_set = CustomDataset(train_dataset)
  val_set =CustomDataset(val_dataset)
  test_set =CustomDataset(test_dataset)

  print('\n Train set :', train_set[0][1] )

  # Undersample the majority class (healthy=1) for both the train datasets
  undersampler_train, undersampler_val= create_weightsamplers(train_set, val_set)

  train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=undersampler_train)#, collate_fn=custom_collate)
  val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, sampler=undersampler_val)#, collate_fn=custom_collate)
  test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)#, collate_fn=custom_collate)


  """# Downloading the ResNet50 model pre-trained on ImageNet
  https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
  """
  # Load the ResNet50 model
  model_base = resnet50(weights=weights)

  # Add more layers and change the last layer to output the number of classes
  num_classes = 1

  model_base.fc = torch.nn.Linear(model_base.fc.in_features, num_classes)

  # Freeze all the layers except the last one
  for param in model_base.parameters():
      param.requires_grad = False

  # Make the last layer trainable
  for param in model_base.fc.parameters():
      param.requires_grad = True
  
  criterion = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

  optimizer = torch.optim.Adam(model_base.parameters(), lr=learning_rate)
  #optimizer = optim.SGD(model_base.parameters(), lr=learning_rate, momentum=0.9)

  # Decay LR by a factor of 0.1 every 5 epochs
  exp_lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

  train_step=make_train_step(model_base, optimizer, criterion)

  model, train_losses, val_losses, train_accuracies, val_accuracies, train_precision, val_precision, train_recall, val_recall, train_roc_auc, val_roc_auc, epochs, misclassified_metadata = train_model(model_base, criterion, optimizer, exp_lr_scheduler, num_epochs)
  test_roc_auc_scores, fpr, tpr, thresh, test_accuracy, test_precision, test_recall, test_misclassified_metadata= test(model)
  
  density_misclassification_rate, zoom_group_misclassification_rate, image_view_misclassification_rate, film_misclassification_rate, digital_misclassification_rate, calcification_misclassification_rate, no_calcification_misclassification_rate, nodule_misclassification_rate, no_nodule_misclassification_rate = misclassification_rates(test_misclassified_metadata, test_df) 
  
  cv_train_accuracies.append(train_accuracies)
  cv_val_accuracies.append(val_accuracies)
  cv_train_precisions.append(train_precision)
  cv_val_precisions.append(val_precision)
  cv_train_recalls.append(train_recall)
  cv_val_recalls.append(val_recall)
  cv_train_roc_auc.append(train_roc_auc)
  cv_val_roc_auc.append(val_roc_auc)
  cv_test_roc_aucs.append(test_roc_auc_scores)
  cv_test_accuracies.append(test_accuracy)
  cv_test_precisions.append(test_precision)
  cv_test_recalls.append(test_recall)
  cv_density_misclassification_rate.append(density_misclassification_rate)
  cv_zoom_group_misclassification_rate.append(zoom_group_misclassification_rate)
  cv_image_view_misclassification_rate.append(image_view_misclassification_rate)
  cv_film_misclassification_rate.append(film_misclassification_rate)
  cv_digital_misclassification_rate.append(digital_misclassification_rate)
  cv_calcification_misclassification_rate.append(calcification_misclassification_rate)
  cv_no_calcification_misclassification_rate.append(no_calcification_misclassification_rate)
  cv_nodule_misclassification_rate.append(nodule_misclassification_rate)
  cv_no_nodule_misclassification_rate.append(no_nodule_misclassification_rate)

  cv_fprs.append(fpr)
  cv_tprs.append(tpr)
  cv_thresh.append(thresh)

  if np.mean(val_roc_auc) > best_val_roc_auc:
    cv_best_model=model
    best_val_roc_auc= np.mean(val_roc_auc)

cv_train_roc_auc = np.array(cv_train_roc_auc)
cv_train_roc_auc_mean = np.mean(cv_train_roc_auc, axis=0)
cv_train_roc_auc_std_dev = np.std(cv_train_roc_auc, axis=0)
print('\nTraining ROC-AUC: ',cv_train_roc_auc_mean, ' +/- ', cv_train_roc_auc_std_dev)

cv_val_roc_auc = np.array(cv_val_roc_auc)
cv_val_roc_auc_mean = np.mean(cv_val_roc_auc, axis=0)
cv_val_roc_auc_std_dev = np.std(cv_val_roc_auc, axis=0)
print('\nValidation ROC-AUC: ',cv_val_roc_auc_mean, ' +/- ', cv_val_roc_auc_std_dev)

cv_test_roc_aucs = np.array(cv_test_roc_aucs)
cv_test_roc_auc_mean = np.mean(cv_test_roc_aucs, axis=0)
cv_test_roc_auc_std_dev = np.std(cv_test_roc_aucs, axis=0)
print('\nTest ROC-AUC: ',cv_test_roc_auc_mean, ' +/- ', cv_test_roc_auc_std_dev)

cv_train_accuracies=np.array(cv_train_accuracies)
cv_train_accuracies_mean = np.mean(cv_train_accuracies)
cv_train_accuracies_std_dev = np.std(cv_train_accuracies)
print("\nTrain accuracy: ", cv_train_accuracies_mean, ' +/- ', cv_train_accuracies_std_dev)

cv_val_accuracies=np.array(cv_val_accuracies)
cv_val_accuracies_mean = np.mean(cv_val_accuracies)
cv_val_accuracies_std_dev = np.std(cv_val_accuracies)
print("\nValidation accuracy: ", cv_val_accuracies_mean, ' +/- ', cv_val_accuracies_std_dev)

cv_test_accuracies=np.array(cv_test_accuracies)
cv_test_accuracies_mean = np.mean(cv_test_accuracies)
cv_test_accuracies_std_dev = np.std(cv_test_accuracies)
print("\nTest accuracy: ", cv_test_accuracies_mean, ' +/- ', cv_test_accuracies_std_dev)

cv_train_precisions = np.array(cv_train_precisions)
# Compute the mean and standard deviation along axis 0
cv_train_precisions_mean = np.mean(cv_train_precisions, axis=0)
cv_train_precisions_std_dev = np.std(cv_train_precisions, axis=0)
print("\nTrain precision: ", cv_train_precisions_mean, ' +/- ', cv_train_precisions_std_dev)

cv_val_precisions = np.array(cv_val_precisions)
# Compute the mean and standard deviation along axis 0
cv_val_precisions_mean = np.mean(cv_val_precisions, axis=0)
cv_val_precisions_std_dev = np.std(cv_val_precisions, axis=0)
print("\nValidation precision: ", cv_val_precisions_mean, ' +/- ', cv_val_precisions_std_dev)

cv_test_precisions = np.array(cv_test_precisions)
# Compute the mean and standard deviation along axis 0
cv_test_precisions_mean = np.mean(cv_test_precisions, axis=0)
cv_test_precisions_std_dev = np.std(cv_test_precisions, axis=0)
print("\nTest precision: ", cv_test_precisions_mean, ' +/- ', cv_test_precisions_std_dev)

cv_train_recalls = np.array(cv_train_recalls)
# Compute the mean and standard deviation along axis 0
cv_train_recalls_mean = np.mean(cv_train_recalls, axis=0)
cv_train_recalls_std_dev = np.std(cv_train_recalls, axis=0)
print("\nTrain recall: ", cv_train_recalls_mean, ' +/- ', cv_train_recalls_std_dev)


cv_val_recalls = np.array(cv_val_recalls)
# Compute the mean and standard deviation along axis 0
cv_val_recalls_mean = np.mean(cv_val_recalls, axis=0)
cv_val_recalls_std_dev = np.std(cv_val_recalls, axis=0)
print("\nValidation recall: ", cv_val_recalls_mean, ' +/- ', cv_val_recalls_std_dev)

cv_test_recalls = np.array(cv_test_recalls)
# Compute the mean and standard deviation along axis 0
cv_test_recalls_mean = np.mean(cv_test_recalls, axis=0)
cv_test_recalls_std_dev = np.std(cv_test_recalls, axis=0)
print("\nTest recall: ", cv_test_recalls_mean, ' +/- ', cv_test_recalls_std_dev)


cv_test_roc_aucs = np.array(cv_test_roc_aucs)
cv_test_roc_aucs_mean = np.mean(cv_test_roc_aucs, axis=0)
cv_test_roc_aucs_std_dev = np.std(cv_test_roc_aucs, axis=0)
print('\nTest ROC-AUC: ',cv_test_roc_aucs_mean,  ' +/- ', cv_test_roc_aucs_std_dev)


if not os.path.exists(save_folder):
    os.makedirs(save_folder)
os.chdir(save_folder)


# Plot accuracies
plt.plot(train_accuracies, label='train accuracy')
plt.plot(val_accuracies, label='val accuracy')
plt.title('Train and Validation Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
plt.savefig('multiclass_accuracies_experiment1.png')
plt.clf()


#Plot ROC curve with standard deviation
mean_fpr = np.linspace(0, 1, 100)  # Points to interpolate the rates
mean_tpr = []
std_tpr = []

from scipy import interp
# Interpolate the rates for each class at the mean_fpr points

tprs = []
for i in range(len(cv_fprs)):
    tprs.append(interp(mean_fpr, cv_fprs[i], cv_tprs[i]))

mean_tpr=np.mean(tprs, axis=0)
std_tpr=np.std(tprs, axis=0)
# Plot ROC curve with standard deviation
plt.figure()
# print('\n mean fpr: ', mean_fpr)
# print('\n mean tpr: ', mean_tpr)
plt.plot(mean_fpr, mean_tpr, label=f'AUC = {cv_test_roc_aucs_mean:.2f} +/- {cv_test_roc_aucs_std_dev:.2f}')
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
# Show the plot
plt.show()

'''Initially a grid search was performed to select the hyperparameters used for the experiments, by default it is not performed anymore'''
GRID_SEARCH=False
if GRID_SEARCH==False:
  '''PRINT THE OUTPUTS IN A .txt FILE IN THE SPECIFIED FOLDER'''
  with open(os.path.join(save_folder,"output.txt"), "w") as file:
      file.write("Problem: '{}'\n".format(task))
      file.write("Number of epochs: '{}'\n".format(epochs))
      file.write("Train and validation zoom groups:\n")
      if filter_zoom_1_train_val: 
        file.write("group 1 \n")
      if filter_zoom_2_train_val: 
        file.write("group 2 \n")
      if filter_zoom_3_train_val: 
        file.write("group 3 \n\n")
      
      file.write("Test zoom groups:\n")
      if filter_zoom_1_test: 
        file.write("group 1 \n")
      if filter_zoom_2_test: 
        file.write("group 2 \n")
      if filter_zoom_3_test: 
        file.write("group 3")

      file.write("\n\nTRAIN      - MALIGN number of samples:{}".format((train_df['classification'] == ' Malign ').sum()))
      file.write("\nVALIDATION - MALIGN number of samples:{}".format((val_df['classification'] == ' Malign ').sum()))
      file.write("\nTEST       - MALIGN number of samples :{}".format((test_df['classification'] == ' Malign ').sum()))
      file.write("\n\nTRAIN      - BENIGN number of samples:{}".format((train_df['classification'] == ' Benign ').sum()))
      file.write("\nVALIDATION - BENIGN number of samples:{}".format((val_df['classification'] == ' Benign ').sum()))
      file.write("\nTEST       - BENIGN number of samples :{}".format((test_df['classification'] == ' Benign ').sum()))
      file.write("\n\nTRAIN      - CALCIFICATION number of samples:{}".format((train_df['calcification'] == 1).sum()))
      file.write("\nVALIDATION - CALCIFICATION number of samples:{}".format((val_df['calcification'] == 1).sum()))
      file.write("\nTEST       - CALCIFICATION number of samples :{}".format((test_df['calcification'] == 1).sum()))
      file.write("\n\nTRAIN      - NODULE number of samples:{}".format((train_df['nodule'] == 1).sum()))
      file.write("\nVALIDATION - NODULE number of samples:{}".format((val_df['nodule'] == 1).sum()))
      file.write("\nTEST       - NODULE number of samples :{}".format((test_df['nodule'] == 1).sum()))

      # Print number of examples in filtered train, validation, and test datasets
      file.write("\n\nNumber of Samples in TRAIN dataset:{}".format(len(train_dataset)))
      file.write("\nNumber of Samples in VALIDATION dataset:{}".format(len(val_dataset)))
      file.write("\nNumber of Samples in TEST dataset:{}".format(len(test_dataset)))

      #roc_auc
      file.write("\n\nTrain ROC-AUC: {} +/- {}".format(cv_train_roc_auc_mean,cv_train_roc_auc_std_dev))
      file.write("\nValidation ROC-AUC: {} +/- {}".format(cv_train_roc_auc_mean,cv_train_roc_auc_std_dev))
      file.write("\nTest ROC-AUC: {} +/- {}".format(cv_test_roc_auc_mean,cv_test_roc_auc_std_dev))
      
      #accuracies
      file.write("\n\nTrain accuracy: {} +/- {}".format(cv_train_accuracies_mean,cv_train_accuracies_std_dev))
      file.write("\nValidation accuracy: {} +/- {}".format(cv_val_accuracies_mean, cv_val_accuracies_std_dev))
      file.write("\nTest accuracy: {} +/- {}\n".format(cv_test_accuracies_mean, cv_test_accuracies_std_dev))
      
      #precision
      file.write("\nTrain precision: {} +/- {}\n".format(cv_train_precisions_mean, cv_train_precisions_std_dev))
      file.write("\nValidation precision: {} +/- {}\n".format(cv_val_precisions_mean, cv_val_precisions_std_dev))
      file.write("\nTest precision: {} +/- {}\n".format(cv_test_precisions_mean, cv_test_precisions_std_dev))
      
      #recall
      file.write("\nTrain recall: {} +/- {}\n".format(cv_train_recalls_mean, cv_train_recalls_std_dev))
      file.write("\nValidation recall: {} +/- {}\n".format(cv_val_recalls_mean, cv_val_recalls_std_dev))
      file.write("\nTest recall: {} +/- {}\n".format(cv_test_recalls_mean, cv_test_recalls_std_dev))


      ##MISCLASSIFICATION RATES: #MISCLASSIFIED SAMPLES FOR THAT CLASS / #TRAINING SAMPLES IN THAT CLASS
      file.write(f'\n\nMISCLASSIFICATION RATES: #MISCLASSIFIED SAMPLES FOR THAT CLASS / #TRAINING SAMPLES IN THAT CLASS')
    
      cv_density_misclassification_rate = np.array(cv_density_misclassification_rate)
      cv_density_misclassification_rate_mean = np.mean(cv_density_misclassification_rate, axis=0)
      cv_density_misclassification_rate_std_dev = np.std(cv_density_misclassification_rate, axis=0)
      output = [f"{cv_density_misclassification_rate_mean[i]:.6f} +/- {cv_density_misclassification_rate_std_dev[i]:.6f}" for i in range(len(cv_density_misclassification_rate_std_dev))]
      file.write("\n\nDensity (1, 2, 3, 4): {}".format(", ".join(output)))

      cv_zoom_group_misclassification_rate = np.array(cv_zoom_group_misclassification_rate)
      cv_zoom_group_misclassification_rate_mean = np.mean(cv_zoom_group_misclassification_rate, axis=0)
      cv_zoom_group_misclassification_rate_std_dev = np.std(cv_zoom_group_misclassification_rate, axis=0)
      output = [f"{cv_zoom_group_misclassification_rate_mean[i]:.6f} +/- {cv_zoom_group_misclassification_rate_std_dev[i]:.6f}" for i in range(len(cv_zoom_group_misclassification_rate_std_dev))]
      file.write("\n\nZoom group: {}".format(", ".join(output)))

      cv_image_view_misclassification_rate = np.array(cv_image_view_misclassification_rate)
      cv_image_view_misclassification_rate_mean = np.mean(cv_image_view_misclassification_rate, axis=0)
      cv_image_view_misclassification_rate_std_dev = np.std(cv_image_view_misclassification_rate, axis=0)
      output = [f"{cv_image_view_misclassification_rate_mean[i]:.6f} +/- {cv_image_view_misclassification_rate_std_dev[i]:.6f}" for i in range(len(cv_image_view_misclassification_rate_std_dev))]
      file.write("\n\nImage view (0, 1, 2, 3, 4): {}".format(", ".join(output)))

      cv_film_misclassification_rate = np.array(cv_film_misclassification_rate)
      cv_film_misclassification_rate_mean = np.mean(cv_film_misclassification_rate, axis=0)
      cv_film_misclassification_rate_std_dev = np.std(cv_film_misclassification_rate, axis=0)
      file.write("\nFilm, {} +/- {}\n".format(cv_film_misclassification_rate_mean,cv_film_misclassification_rate_std_dev))
      
      cv_digital_misclassification_rate = np.array(cv_digital_misclassification_rate)
      cv_digital_misclassification_rate_mean = np.mean(cv_digital_misclassification_rate, axis=0)
      cv_digital_misclassification_rate_std_dev = np.std(cv_digital_misclassification_rate, axis=0)
      file.write("\nDigital, {} +/- {}\n".format(cv_digital_misclassification_rate_mean,cv_digital_misclassification_rate_std_dev))
      
      cv_no_calcification_misclassification_rate = np.array(cv_no_calcification_misclassification_rate)
      cv_no_calcification_misclassification_rate_mean = np.mean(cv_no_calcification_misclassification_rate, axis=0)
      cv_no_calcification_misclassification_rate_std_dev = np.std(cv_no_calcification_misclassification_rate, axis=0)
      file.write("\nNo calcification, {:.4f} +/- {:.4f}\n".format(cv_no_calcification_misclassification_rate_mean,cv_no_calcification_misclassification_rate_std_dev))
      
      cv_calcification_misclassification_rate = np.array(cv_calcification_misclassification_rate)
      cv_calcification_misclassification_rate_mean = np.mean(cv_calcification_misclassification_rate, axis=0)
      cv_calcification_misclassification_rate_std_dev = np.std(cv_calcification_misclassification_rate, axis=0)
      file.write("\nCalcification, {:.4f} +/- {:.4f}\n".format(cv_calcification_misclassification_rate_mean,cv_calcification_misclassification_rate_std_dev))
      
      cv_no_nodule_misclassification_rate = np.array(cv_no_nodule_misclassification_rate)
      cv_no_nodule_misclassification_rate_mean = np.mean(cv_no_nodule_misclassification_rate, axis=0)
      cv_no_nodule_misclassification_rate_std_dev = np.std(cv_no_nodule_misclassification_rate, axis=0)
      file.write("\nNo nodule, {:.4f} +/- {:.4f}\n".format(cv_no_nodule_misclassification_rate_mean,cv_no_nodule_misclassification_rate_std_dev))
      
      cv_nodule_misclassification_rate = np.array(cv_nodule_misclassification_rate)
      cv_nodule_misclassification_rate_mean = np.mean(cv_nodule_misclassification_rate, axis=0)
      cv_nodule_misclassification_rate_std_dev = np.std(cv_nodule_misclassification_rate, axis=0)
      file.write("\nNodule, {:.4f} +/- {:.4f}\n".format(cv_nodule_misclassification_rate_mean,cv_nodule_misclassification_rate_std_dev))
      

