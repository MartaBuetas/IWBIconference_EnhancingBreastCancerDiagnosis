'''Create a .jsonl file for the metadata of the patches from the .csv files generated with the code 'generate_patch_dataset.py'''


import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
from torchvision import datasets, transforms
import seaborn as sns
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# # Creating metadata:

# First, reading the csv files (fixing one problem with healthy_patch_features.csv file, but it is now solved)

root=input("Path of the metadata .csv files generated when the patch dataset was created: ")
os.chdir(root)

# Read the corresponding .csv file
df_healthy_digital= pd.DataFrame()
df_healthy_digital_path=os.path.join(root, 'digital_healthy_patch_features.csv') 
df_healthy_digital = pd.read_csv(df_healthy_digital_path)
df_healthy_digital.rename(columns={'image_type_id':'image_view'},  inplace = True)
df_healthy_digital['classification']='Normal'
df_healthy_digital['age']=df_healthy_digital['age'].astype(int)

df_healthy_digital['zoom_group']=0

# Read the corresponding .csv file
df_lesion= pd.DataFrame()
df_lesion_path=os.path.join(root, 'lesion_patch_features.csv') 
df_lesion = pd.read_csv(df_lesion_path)

df_lesion['age']=df_lesion['age'].replace('\s+', '', regex=True)
df_lesion['age']=pd.to_numeric(df_lesion['age'], errors='coerce')
df_lesion['age']=df_lesion['age'].fillna(df_lesion['age'].mean())
df_lesion['age']=df_lesion['age'].astype(int)

df_lesion['zoom_group']=df_lesion['zoom_group'].astype(int)

# Read the corresponding .csv file
df_healthy_film= pd.DataFrame()
df_healthy_film_path=os.path.join(root, 'film_healthy_patch_features.csv') 
df_healthy_film = pd.read_csv(df_healthy_film_path)
df_healthy_film['age']=df_healthy_film['age'].replace('\s+', '', regex=True)
df_healthy_film['age']=pd.to_numeric(df_healthy_film['age'], errors='coerce')
df_healthy_film['age']=df_healthy_film['age'].fillna(df_lesion['age'].mean())
df_healthy_film['age']=df_healthy_film['age'].astype(int)
df_healthy_film['zoom_group']=0

# Joining the three dataframes:

df_lesion=df_lesion[['patch_id', 'patient_id', 'patch_file_name', 'image_view', 'age', 'density', 'Format', 'classification', 'mammography_nodule', 'mammography_calcification', 'zoom_group']]
df_lesion['suspicious']=1
df_lesion.rename(columns={'mammography_nodule':'nodule'},  inplace = True)
df_lesion.rename(columns={'mammography_calcification':'calcification'},  inplace = True)
df_healthy_digital=df_healthy_digital[['patch_id', 'patient_id', 'patch_file_name', 'image_view', 'age', 'density', 'Format', 'classification', 'zoom_group']]
df_healthy_digital['suspicious']=0
df_healthy_digital['nodule']=0
df_healthy_digital['calcification']=0
df_healthy_digital['classification']='Normal'
df_healthy_film=df_healthy_film[['patch_id', 'patient_id', 'patch_file_name', 'image_view', 'age', 'density', 'Format', 'classification', 'zoom_group']]
df_healthy_film['suspicious']=0
df_healthy_film['nodule']=0
df_healthy_film['calcification']=0
df_healthy_film['classification']='Normal'

df_metadata = pd.concat([df_lesion, df_healthy_digital, df_healthy_film])


df_metadata=df_metadata.reset_index()
del df_metadata['index']

df_metadata = df_metadata.sort_values(by=['patch_id'], ascending=True)

df_metadata['image_view']=df_metadata['image_view'].astype(int)

df_metadata['density']=df_metadata['density'].replace('\s+', '', regex=True)
df_metadata['density']=pd.to_numeric(df_metadata['density'], errors='coerce')
df_metadata['density']=df_metadata['density'].fillna(df_metadata['density'].mean())
df_metadata['density']=df_metadata['density'].astype(int)

df_metadata['zoom_group']=df_metadata['zoom_group'].astype(int)

df_metadata.rename(columns={'patch_file_name':'file_name'},  inplace = True)
df_metadata

# Create a metadata.jsonl file:

# Select the columns to include in the metadata file
selected_columns = ['file_name', 'patient_id', 'image_view', 'Format', 'classification', 'age', 'density', 'zoom_group', 'suspicious', 'calcification', 'nodule']

# Iterate over each row in the dataframe, and create a dictionary with the selected columns
metadata = []
for _, row in df_metadata.iterrows():
    metadata_dict = {}
    for col in selected_columns:
        metadata_dict[col] = row[col]
    metadata.append(metadata_dict)

# Write each dictionary as a JSON object in a new line in the metadata file
with open('metadata.jsonl', 'w') as f:
    for m in metadata:
        json.dump(m, f)
        f.write('\n')

# Delete those patches with no metadata:

image_dir = os.path.join(root, 'data')

# Open the metadata file and read the list of filenames with metadata
metadata_filenames = []
with open('metadata.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['file_name'] in os.listdir(image_dir):
            metadata_filenames.append(data['file_name'])
        else:
            print(data['file_name'])
print(len(metadata_filenames))

count=0
# Filter the image files based on the list of filenames with metadata

for filename in os.listdir(image_dir):
    if filename not in metadata_filenames:
        print(filename)
        os.remove(os.path.join(image_dir, filename))
        count=count+1

print(count, ' patches were deleted.')

# Load the metadata from the JSONL file
metadata_path=os.path.join(root, 'metadata.jsonl')
with open(metadata_path, "r") as f:
    metadata = [json.loads(line) for line in f]