from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import config
import os

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data

def random_translation(data1):
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    return data1[10+i:170+i,18+j:210+j,10+z:170+z]

def crop(data1):
    return data1[10:170,18:210,10:170]

# This is for the training of the first stage
class OneDataset(Dataset):
    def __init__(self, root_Abeta = config.whole_Abeta, task = config.train, stage = "train"):
        self.root_Abeta = root_Abeta
        self.task = task
        self.images = read_list(self.task)
        self.length_dataset = len(self.images)
        self.len = len(self.images)
        self.stage = stage

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        name = self.images[index % self.len] + ".nii"
        path_Abeta = os.path.join(self.root_Abeta, name)
        Abeta = nifti_to_numpy(path_Abeta)
        if self.stage == "train":
            Abeta = random_translation(Abeta)
        else:
            Abeta = crop(Abeta)
        return Abeta, name

# This is for the training of the second stage
class TwoDataset(Dataset):
    def __init__(self,root_MRI = config.whole_MRI, root_Abeta = config.whole_Abeta, task = config.train, stage = "train"):
        self.root_Abeta = root_Abeta
        self.root_MRI = root_MRI
        self.task = task
        self.images = read_list(self.task)
        self.length_dataset = len(self.images)
        self.len = len(self.images)
        self.stage = stage

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        name = self.images[index % self.len] + ".nii"
        path_Abeta = os.path.join(self.root_Abeta, name)
        Abeta = nifti_to_numpy(path_Abeta)
        path_MRI = os.path.join(self.root_MRI, name)
        MRI = nifti_to_numpy(path_MRI)
        MRI = crop(MRI)
        if self.stage != "train":
            Abeta = crop(Abeta)
        data = pd.read_csv("data_info/data_info.csv",encoding = "ISO-8859-1")
        label = data[data['ID'] == name[0:-4]]['label']
        label=label.values
        label=label.astype(np.float32)
        return MRI, Abeta, name, label