import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tutils
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import pandas as pd
# from focal_loss_pytorch.focalloss import FocalLoss
from pytorch_workplace.focalloss import loss as FocalLoss
from imet_data_loader import IMetDataset
from train_utils import *
import argparse

parser = argparse.ArgumentParser(description='Train iMet classifier')
parser.add_argument('--data_dir', type=str, default='/raid/hlcv19/team22/imet/data/', help='iMet dataset location')
parser.add_argument('--labels_file', type=str, help='Labels file name', default='labels.csv')
parser.add_argument('--train_file', type=str, help='Train file name', default='train_.csv')
parser.add_argument("--weighted_classes", type=bool, default=False)
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--focal_loss", type=bool, default=False)
parser.add_argument("--positive_weight", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--load_model", type=str, default='/raid/hlcv19/team22/imet/data/resnet18_tr__0.9_focal_gamma2')
parser.add_argument("--save_model", type=str, default='/raid/hlcv19/team22/imet/data/resnet18')
parser.add_argument("--convnet_type", type=str, default='resnet18')

args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
   
# --------------------------------
# Hyper-parameters
# --------------------------------
data_dir = args.data_dir
dir_path = os.path.join(data_dir, 'train')
labels_path = os.path.join(data_dir, args.labels_file)
annotations_path = os.path.join(data_dir, args.train_file)

train_data = IMetDataset(root_dir=dir_path, labels_csv=labels_path,
                         annotations_csv=annotations_path, 
                         transform=train_data_transforms)


num_epochs = args.epochs
batch_size = args.batch_size
cosine_padded_len = 15
pretrained = True
learning_rate = 0.01
dataset_size = len(train_data)
num_training = int(dataset_size * 0.6)
num_validation = int(dataset_size * 0.2)
dataset_size = {'train': num_training, 'val': num_validation}
fine_tune = True
num_classes = len(train_data.labels_frame)

val_data = IMetDataset(root_dir=dir_path, labels_csv=labels_path, 
                         annotations_csv=annotations_path,
                       transform=val_data_transforms)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)


mask = list(range(num_training))
train_dataset = tutils.dataset.Subset(train_data, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.dataset.Subset(val_data, mask)
mask = list(range(num_training + num_validation, num_training + num_validation*2))
test_dataset = torch.utils.data.dataset.Subset(val_data, mask)


# -------------------------------------------------
# Data loader
# -------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)

loader = {'train': train_loader, 'val': val_loader}
print("loaders done")
    
def train_model(model, optimizer, criterion_classification,  scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f2 = 0.0
    best_loss = 999.0
    history = []
    
    threshold = 0.5
    for epoch_i in range(num_epochs):
        print('Epoch {}/{}'.format(epoch_i, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = ['val']
        if args.train:
            phases = ['train'] + phases
        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss_embedding = 0.0
            running_loss_classification = 0.0
            running_corrects = 0
            score_num = 0.
            preds = []
            targets = []
            
            # Iterate over data.
            for inputs, labels in loader[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output_classes = model(inputs)
                    loss_classifier = criterion_classification(output_classes.double(), labels)
                    preds.append(torch.sigmoid(output_classes).clone())
                    targets.append(labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss_classifier
                        loss.backward()
                        optimizer.step()
                        
                    score_num = score_num + 1
                    
                if score_num % (5000/batch_size) == 0:
                    print('{}  Loss classification: {:.4f}'.format(
                        score_num, loss_classifier))
                # statistics
                running_loss_classification += loss_classifier.item() * inputs.size(0)
            
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            if phase == 'train':
                threshold = find_threshold(preds, targets.int())
            f2, prec, recall, acc = f2score((preds>threshold).int(), targets.int(), ret_pr=True)
            epoch_loss_classification = running_loss_classification / dataset_size[phase]

            print('{} Loss classification: {:.4f}, F2-score {:.4f}, precision {:.4f}, recall {:.4f} acc {:.4f}'.format(
                phase, epoch_loss_classification, f2.item(), prec.item(), recall.item(), acc.item()))

            history.append((phase, epoch_loss_classification, f2, prec, recall,))
            # deep copy the model
            if phase == 'val' and f2 > best_f2:
                best_f2 = f2
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, args.save_model)
                print('model saved')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(history)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

print("using weighted classification {}".format(args.weighted_classes))

pw = torch.tensor([args.positive_weight]*num_classes, device=device, dtype=torch.double)
criterion_classification =  nn.BCEWithLogitsLoss(pos_weight=pw)
if args.weighted_classes:
    criterion_classification =  nn.BCEWithLogitsLoss(train_data.getLabelWeights(), pos_weight=pw)
if args.focal_loss:
    gamma=2
    criterion = FocalLoss.FocalLoss(gamma=gamma)

if args.load_model:
    model = torch.load(args.load_model)
else:
    if args.convnet_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    if args.convnet_type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    for param in model.parameters():
        param.requires_grad = True
print("model load done")

optimizer_ft = optim.Adam(model.parameters())
print("optimizer done")

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print("lr done")

model = model.to(device)
model, threshold = train_model(model, optimizer_ft, criterion_classification, exp_lr_scheduler, num_epochs=num_epochs)
model_eval(model, threshold, test_loader)
print("train done")
torch.save(model, os.path.join(data_dir, args.load_model))
