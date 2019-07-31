import torch
import pandas as pd
import numpy as np 
from skimage import io
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IMetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_csv, annotations_csv, transform=None):
        super(IMetDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.labels_frame = pd.read_csv(labels_csv)
        self.annotations_frame = pd.read_csv(annotations_csv)\
        
        labels = [[int(x) for x in s.split(' ')] for s in self.annotations_frame.iloc[:, 1]]
        flat_labels = []
        for l in labels:
              flat_labels.extend(l)
        self.frequencies = np.unique(flat_labels, return_counts=True)[1]
        self.label_weights = 1/self.frequencies
        
    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        img_name = self.annotations_frame.iloc[idx, 0]
        path = os.path.join(self.root_dir, img_name) + '.png'
        image = io.imread(path)
        image = np.asarray(image)
        label_idxs = self.annotations_frame.iloc[idx, 1]
        label_idxs = np.array(label_idxs.split(' ')).astype(np.int)
        labels = np.zeros((len(self.labels_frame)))
        labels[label_idxs] = 1
        labels = torch.DoubleTensor(labels)#*torch.cutoff_mask
        if self.transform:
            image = self.transform(image)
        return image, labels
    
    def getCutoffMask(self, freq):
        return self.frequencies >= freq
    
    def getLabelWeights(self):
        return self.label_weights

    def split_labels(self, v):
        v=v.split(' ')
        v = np.asarray([np.int(l) for l in v])
        labels = np.zeros((len(self.labels_frame)))
        labels[v]=1
        return labels