task='Multiclass'
num_singans=4
num_folds = 3

## IMPORTS
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
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
from datasets import load_dataset, concatenate_datasets
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
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, roc_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

"""# Load data"""
root='/home/marta/data'
dataset = load_dataset("imagefolder", data_dir=root)
oversampling_model=input("Which experiment do you want to test?\n- Option 1: Model trained with oversampled set and no synthetic images\n- Option 2: Model trained with augmented set using SinGAN models\n- Option 3: Ensemble architecture using models from options 1 and 2 \n- Option 4: Baseline without data augmentation. \n[type '1', '2', '3' or '4']: ")

if int(oversampling_model) == 1:
   results_folder = 'iwbi_oversampling_z1_train_3'
elif int(oversampling_model) == 2:
   results_folder = 'iwbi_4singan_z1_train_3'
elif int(oversampling_model) == 3:
   results_folder = ['iwbi_4singan_z1_train_3', 'iwbi_oversampling_z1_train_3']# Perform filtering based on user input for test set
elif int(oversampling_model) == 4:
   results_folder = 'iwbi_baseline_train_z3'

"""OPTIONAL ZOOM GROUP FILTER"""
filter_zoom_1_test = input("Filter Zoom Group 1 for test set? (y/n): ").lower() == 'y'
filter_zoom_2_test = input("Filter Zoom Group 2 for test set? (y/n): ").lower() == 'y'
filter_zoom_3_test = input("Filter Zoom Group 3 for test set? (y/n): ").lower() == 'y'

'''OPTIONAL RANDOM RESIZE CROP'''
random_resize = input("Do you want to implement a random resize crop? (y/n): ").lower() == 'y'

"""Reading metadata"""

# Load the metadata from the JSONL file (which is inside of the 'data' folder)
metadata_path=os.path.join(root, 'metadata.jsonl')
with open(metadata_path, "r") as f:
    metadata = [json.loads(line) for line in f]

# Convert the list of dictionaries into a pandas dataframe
metadata_df = pd.DataFrame([sample for sample in metadata])

def dfToDict(df):
  '''Given a single row dataframe, it returns a dictionary'''
  dict={'file_name': df['file_name'].item(),	
   'patient_id': df['patient_id'].item(),	
   'image_view': df['image_view'].item(), 
   'Format': df['Format'].item(), 
   'classification': df['classification'].item(), 
   'age': df['age'].item(), 
   'density': df['density'].item(), 
   'zoom_group': df['zoom_group'].item(), 
   'suspicious': df['suspicious'].item(), 
   'calcification': df['calcification'].item(), 
   'nodule': df['nodule'].item()}	
  return dict


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
#  Extract the labels and filenames from the metadata
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

print('\n\n# of unique patients: ',len(set(patients)))

"""As the patient_ids have to be integers in order to be able to handle them as a group for the splitter, I will map each patient_id to an unique integer."""

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

# I read the test indices saved to use the same accross every experiment
with open('indices_test.txt', 'r') as file:
    test_indices = [int(line.strip()) for line in file]

# The rest of the indices are train_val_indices
test_indices = np.array(test_indices, dtype=np.int64)

zoom_groups = data['train']["zoom_group"]
zoom_groups = np.asarray(zoom_groups)

"""FILTER BY GROUP OF ZOOM"""
test_indices = test_indices[((zoom_groups[test_indices] == 1) & filter_zoom_1_test) |
                         ((zoom_groups[test_indices] == 2) & filter_zoom_2_test) |
                         ((zoom_groups[test_indices] == 3) & filter_zoom_3_test) |
                         (zoom_groups[test_indices] == 0)]

test_dataset = data['train'].select(test_indices)

BATCH_SIZE=128

#Load ResNet50 trained weights on ImageNet 
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

if random_resize:
  # Some needed transforms
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      transforms.Lambda(lambda x: preprocess(x))
  ])
else:
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
        self.nodule = dataset['nodule']
        self.calcification = dataset['calcification']
        self.Format = dataset['Format']
        self.zoom_group = dataset['zoom_group']
        self.density = dataset['density']
        self.classification = dataset['classification']
        self.image_view = dataset['image_view']

        # The label is a 3-d vector for the following three classes
        self.class_mapping = {' Malign ': torch.tensor([1, 0]),
                              ' Benign ': torch.tensor([0, 1]),
                              'Normal': torch.tensor([0,0])}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.convert('RGB')
        image = transform(image)
        
        one_hot = self.class_mapping[self.classification[idx]]

        labels = torch.cat((torch.tensor([self.healthy[idx]]), one_hot), dim=0)

        metadata = {
            'format': self.Format[idx],
            'zoom_group': self.zoom_group[idx],
            'density': self.density[idx],
            'classification': self.classification[idx],
            'image_view': self.image_view[idx],
            'calcification': self.calcification[idx],
            'nodule':self.nodule[idx]
        }
        
        return image, labels, metadata
    
test_set =CustomDataset(test_dataset)
print('\n Metadata test set : ', test_set.zoom_group)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

'''Test function'''
def test(test_model):
    test_model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    correct_indices=[]
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, m_batch = data
            outputs = test_model(inputs)

            targets = targets.to(torch.float32)
            outputs = outputs.softmax(dim=-1)
            
            # Targets
            y_true = torch.cat((y_true, targets), 0)
            # Predictions
            y_score = torch.cat((y_score, outputs), 0)
            
            predicted = torch.argmax(outputs, dim=1)
            labels = torch.argmax(targets, dim=1)
            correct_mask = labels == predicted
            correct_indices.append(correct_mask.tolist())
            
            '''Extract metadata info of misclassified samples'''  
            misclassified_indices = correct_mask == False
            misclassified_metadata_batch = {key: [test[i] for i, misclass in enumerate(misclassified_indices) if misclass] for key, test in m_batch.items()}
            if i==0:
                misclassified_metadata=misclassified_metadata_batch
            else:
              for key in misclassified_metadata:
                misclassified_metadata[key].extend(misclassified_metadata_batch[key])

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
    return y_true, y_score, misclassified_metadata, correct_indices

def load_trained_model(current_fold, experiment_folder):

    trained_model = resnet50(weights=weights)
    # Add more layers and change the last layer to output the number of classes
    num_classes = 3
    trained_model.fc = torch.nn.Linear(trained_model.fc.in_features, num_classes)
    # Freeze all the layers except the last one
    for param in trained_model.parameters():
        param.requires_grad = False
    # Make the last layer trainable
    for param in trained_model.fc.parameters():
        param.requires_grad = True

    trained_model_filepath= os.path.join(experiment_folder, 'trained_models' )
    os.chdir(trained_model_filepath)
    print('\n === CURRENT FOLD VARIABLE ===', str(current_fold))
    # To load the trained models: 
    trained_model.load_state_dict(torch.load(str(trained_model_filepath) +'/trained_model_fold_' +str(current_fold+1) +'.pth'))
    return trained_model

cv_test_roc_aucs=[]
cv_test_precisions=[]
cv_test_accuracies=[]
cv_test_recalls=[]

n_classes = 3
cv_fprs = [[] for _ in range(n_classes)]
cv_tprs = [[] for _ in range(n_classes)]
cv_thresh = [[] for _ in range(n_classes)]

if int(oversampling_model) == 2 or int(oversampling_model) == 1  or int(oversampling_model) == 4:
  # Go to the experiment folder where the trained models are saved:
  experiment_folder = os.path.join('/home/marta/iwbi', results_folder)
  os.chdir(experiment_folder)
  ### TEST FOR EACH FOLD
  y_pred_test_folds=[]

  for current_fold in range(num_folds):
      # Read model
      trained_model = load_trained_model(current_fold, experiment_folder)

      # Obtain predictions
      y_true_test, y_pred_test, test_misclassified_metadata, test_correct_indices= test(trained_model)
      y_pred_test_folds.append(y_pred_test) # Save predictions of this fold

      print('==> Evaluating model ...')
      #Print roc_auc socres:
      test_roc_auc_scores = []
      fpr = {}
      tpr = {}
      thresh = {}

      # Roc - auc for each class
      for i in range(y_true_test.shape[1]):
          roc_auc = roc_auc_score(y_true_test[:, i], y_pred_test[:, i])
          test_roc_auc_scores.append(roc_auc)
          fpr[i], tpr[i], thresh[i] = roc_curve(y_true_test[:, i], y_pred_test[:, i])
      
      # Convert to class labels
      y_true_labels = np.argmax(y_true_test, axis=1)
      y_pred_labels = np.argmax(y_pred_test, axis=1)

      # Compute accuracy
      test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
      # Compute precision, recall, and F1-score
      test_precision, test_recall, _, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average=None)
  
      cv_test_roc_aucs.append(test_roc_auc_scores)
      cv_test_accuracies.append(test_accuracy)
      cv_test_precisions.append(test_precision)
      cv_test_recalls.append(test_recall)
      for i in range(len(fpr)):  
        cv_fprs[i].append(fpr[i])
        cv_tprs[i].append(tpr[i])
        cv_thresh[i].append(thresh[i])
  ## AVERAGE OF ROC AUC FOR THE INT(NUM_FOLDS) MODELS:
  cv_test_roc_aucs = np.array(cv_test_roc_aucs)
  cv_test_roc_auc_mean = np.mean(cv_test_roc_aucs, axis=0)
  cv_test_roc_auc_std_dev = np.std(cv_test_roc_aucs, axis=0)
  output = [f"{cv_test_roc_auc_mean[i]:.6f} +/- {cv_test_roc_auc_std_dev[i]:.6f}" for i in range(len(cv_test_roc_auc_std_dev))]
  print('\nTest ROC-AUC: ',output) 

  y_pred_test_folds = np.array(y_pred_test_folds)
  # Mean of predictions accross every fold:
  y_pred_test_folds_mean = np.mean(y_pred_test_folds, axis = 0)
  #ROC-AUC computed for the average predictions of this model
  test_roc_auc_scores=[]
  for i in range(y_true_test.shape[1]):
    roc_auc = roc_auc_score(y_true_test[:, i], y_pred_test_folds_mean[:, i])
    test_roc_auc_scores.append(roc_auc)
  print("\n ROC-AUC for model ", results_folder, " computed from the average predictions: ", test_roc_auc_scores)

else:
  num_classes=3
  y_pred_test_models = []
  for results_folder_i in results_folder: 
    experiment_folder = os.path.join('/home/marta/iwbi', results_folder_i)
    os.chdir(experiment_folder)
    print('\n === Trained model folder: ', results_folder_i)
    y_pred_test_folds =[]
    y_pred_test_folds_mean = []
    
    for current_fold in range(num_folds):
      # Read model
      trained_model = load_trained_model(current_fold, experiment_folder)

      # Obtain predictions
      y_true_test, y_pred_test, test_misclassified_metadata, test_correct_indices= test(trained_model)
      y_pred_test_folds.append(y_pred_test) # Save predictions of this fold

      test_roc_auc_scores=[]
      for i in range(y_true_test.shape[1]):
        roc_auc = roc_auc_score(y_true_test[:, i], y_pred_test[:, i])
        test_roc_auc_scores.append(roc_auc)
      
      print('\n Rocauc for fold ', current_fold, ' for model ', results_folder_i, ': ', test_roc_auc_scores)

    y_pred_test_folds = np.array(y_pred_test_folds)
    # print('\n y_pred_test_folds[0][0] are the predictions for the first sample in the first fold', y_pred_test_folds[0][0])
    # print('\n y_pred_test_folds[1][0] are the predictions for the first sample in the second fold', y_pred_test_folds[1][0])
    # print('\n y_pred_test_folds[2][0] are the predictions for the first sample in the third fold', y_pred_test_folds[2][0])
    # print('\nLen y_pred_test_folds: ', len(y_pred_test_folds))
    # print('\nLen y_pred_test_folds[0], should be the len(test): ', len(y_pred_test_folds[0]))
    # Mean of predictions accross every fold:
    y_pred_test_folds_mean = np.mean(y_pred_test_folds, axis = 0)
    # print('\n y_pred_test_folds_mean[0] are the average predictions for the first sample across the three folds', y_pred_test_folds_mean[0])
    
    y_pred_test_folds_std=np.std(y_pred_test_folds, axis=0)
    # print('\n y_pred_test_folds_std', y_pred_test_folds_std)

    # print('\nLen y_pred_test_folds_mean, should be the len(test): ', len(y_pred_test_folds_mean))

    # Mean of predictions accross the 3 folds
    y_pred_test_models.append(y_pred_test_folds_mean)
    # y_pred_test_models_std.append(y_pred_test_folds_std)

    #ROC-AUC computed for the average predictions of this model
    test_roc_auc_scores=[]
    for i in range(y_true_test.shape[1]):
      roc_auc = roc_auc_score(y_true_test[:, i], y_pred_test_folds_mean[:, i])
      test_roc_auc_scores.append(roc_auc)
    print("\n ROC-AUC for model ", results_folder_i, " computed from the average predictions: ", test_roc_auc_scores)

  # Predictions per model to ensemble them with the average
  y_pred_test_models = np.array(y_pred_test_models)
  # print('\n y_pred_test_models[0][0] are the predictions for the first sample in the first model', y_pred_test_models[0][0])
  # print('\n y_pred_test_models[1][0] are the predictions for the first sample in the second model', y_pred_test_models[1][0])
    
  # Mean of predictions across both models:
  y_pred_test_models_mean = np.mean(y_pred_test_models, axis = 0)
  # print('\n y_pred_test_models_mean[0] are the average predictions for the first sample in the two models', y_pred_test_models_mean[0])
  y_pred_test_models_std=np.std(y_pred_test_models, axis=0)
  #print('\nLen y_pred_test_models_mean, should be the len(test): ', len(y_pred_test_models_mean))
  #print('\n y_pred_test_models_std', y_pred_test_models_std)
  print('==> Evaluating model ...')
  #Print roc_auc socres:

  test_roc_auc_scores = []
  fpr = {}
  tpr = {}
  thresh = {}

  # Roc - auc for each class computed from the average predictions of both models
  for i in range(y_true_test.shape[1]):
    roc_auc = roc_auc_score(y_true_test[:, i], y_pred_test_models_mean[:, i])
    test_roc_auc_scores.append(roc_auc)
    # fpr[i], tpr[i], thresh[i] = roc_curve(y_true_test[:, i], y_pred_test_models_mean[:, i])
  test_roc_auc_mean = np.mean(test_roc_auc_scores)
  print('\n ROC-AUC computed from the average predictions across both models: ', test_roc_auc_scores)

  #print('\n y_true_test ', y_true_test)
  # Convert to class labels
  y_true_labels = np.argmax(y_true_test, axis=1)
  y_pred_labels = np.argmax(y_pred_test, axis=1)

  # Compute accuracy
  test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
  # Compute precision, recall, and F1-score
  test_precision, test_recall, _, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average=None)

  # cv_test_roc_aucs.append(test_roc_auc_scores)
  cv_test_accuracies.append(test_accuracy)
  cv_test_precisions.append(test_precision)
  cv_test_recalls.append(test_recall)
  for i in range(len(fpr)):  
    cv_fprs[i].append(fpr[i])
    cv_tprs[i].append(tpr[i])
    cv_thresh[i].append(thresh[i])
  ## AVERAGE OF ROC AUC FOR THE INT(NUM_FOLDS) MODELS:
  # cv_test_roc_aucs = np.array(cv_test_roc_aucs)
  # cv_test_roc_auc_mean = np.mean(cv_test_roc_aucs, axis=0)
  # cv_test_roc_auc_std_dev = np.std(cv_test_roc_aucs, axis=0)
  # output = [f"{cv_test_roc_auc_mean[i]:.6f} +/- {cv_test_roc_auc_std_dev[i]:.6f}" for i in range(len(cv_test_roc_auc_std_dev))]
  # print('\nTest ROC-AUC: ',test_roc_auc_scores)






 ### AVERAGE OF THE PREDICTIONS OF THE INT(NUM_FOLDS) MODELS: