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
from label_embeddings import LabelEmbeddings
from train_utils import *
from image_embedding_models import ImageEmbeddingClassifier, ImageEmbeddingClassifierDistance
import argparse

parser = argparse.ArgumentParser(description='Train iMet classifier')
parser.add_argument('--data_dir', type=str, default='/raid/hlcv19/team22/imet/data/', help='iMet dataset location')
parser.add_argument('--labels_file', type=str, help='Labels file name', default='labels.csv')
parser.add_argument('--train_file', type=str, help='Train file name', default='train_.csv')
parser.add_argument('--img_emb_model', type=str, help='Image embeddings model', default='/raid/hlcv19/team22/imet/data/resnet18_embeddings_trainable_resnet_2.0')
parser.add_argument("--fix_convnet", type=bool, default=True)
parser.add_argument("--dist_classifier", type=bool, default=False)
parser.add_argument("--train_embeddings", type=bool, default=False)
parser.add_argument("--weighted_classes", type=bool, default=False)
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--focal_loss", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--load_model", type=str)
parser.add_argument("--label2v_model", type=str, default='/raid/hlcv19/team22/imet/data/label2v.model')
parser.add_argument("--save_model", type=str, default='/raid/hlcv19/team22/imet/data/resnet18_embeddings_with_classification')

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

#if not args.img_emb_model:
label_embeddings = LabelEmbeddings(labels_path, annotations_path, load_path=args.label2v_model)
#else:
imem_model = torch.load(args.img_emb_model)
#    label_embeddings = imem_model.label_embeddings
    
print("labels done")
embedding_dims = label_embeddings.dim()

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
fix_resnet = args.fix_convnet
keep_training_embeddings = args.train_embeddings
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
    
def train_model(model, optimizer, criterion_embeddings, criterion_classification, scheduler, keep_training_embeddings, num_epochs):
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
                embeddings, y = onehot2vectors(labels, label_embeddings, padded_len=cosine_padded_len)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output_classes, output_embedding = model(inputs)
                    output_embeddings = output_embedding.double().unsqueeze(1).repeat(1, cosine_padded_len, 1)
                    loss_embedding = criterion_embeddings(output_embeddings.view(-1, embedding_dims), embeddings.view(-1, embedding_dims), y.view(-1))
                    loss_classifier = criterion_classification(output_classes.double(), labels)
                    preds.append(torch.sigmoid(output_classes).clone())
                    targets.append(labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss_classifier
                        if keep_training_embeddings:
                            loss += loss_embedding
                        loss.backward()
                        optimizer.step()
                        
                    score_num = score_num + 1
                    
                if score_num % (5000/batch_size) == 0:
                    print('{} Loss embedding: {:.4f}, Loss classification: {:.4f}'.format(
                        score_num, loss_embedding, loss_classifier))
                # statistics
                running_loss_embedding += loss_embedding.item() * inputs.size(0)
                running_loss_classification += loss_classifier.item() * inputs.size(0)
            
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            if phase == 'train':
                threshold = find_threshold(preds, targets.int())
            f2, prec, recall, acc = f2score((preds>threshold).int(), targets.int(), ret_pr=True)
            epoch_loss_embedding = running_loss_embedding / dataset_size[phase]
            epoch_loss_classification = running_loss_classification / dataset_size[phase]

            print('{} Loss embedding: {:.4f}, Loss classification: {:.4f}, F2-score {:.4f}, precision {:.4f}, recall {:.4f}, acc {:.4f} '.format(
                phase, epoch_loss_embedding, epoch_loss_classification, f2.item(), prec.item(), recall.item(), acc.item()))
            history.append((phase, epoch_loss_embedding, epoch_loss_classification, f2, prec, recall, acc))
            # deep copy the model
            if phase == 'val' and best_f2 > f2:
                best_loss = epoch_loss_embedding
                best_f2 = f2
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, args.save_model)
                print('model saved')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, threshold

print("using weighted classification {}".format(args.weighted_classes))

criterion_embeddings = nn.CosineEmbeddingLoss()
criterion_classification =  nn.BCEWithLogitsLoss()
if args.weighted_classes:
    criterion_classification =  nn.BCEWithLogitsLoss(train_data.getLabelWeights())
gamma=2
if args.focal_loss:
    criterion = FocalLoss.FocalLoss(gamma=gamma)

if args.load_model:
    model = torch.load(args.load_model)
else:
    if args.dist_classifier:
        model = ImageEmbeddingClassifierDistance(imem_model, num_classes, label_embeddings, fix_convnet=args.fix_convnet)
    else:
        model = ImageEmbeddingClassifier(imem_model, num_classes, label_embeddings, fix_convnet=args.fix_convnet)
print("model load done")


# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.4)
#optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
optimizer_ft = optim.Adam(model.parameters())
print("optimizer done")

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print("lr done")

model = model.to(device)
print("keep training embeddings {}".format(keep_training_embeddings))
model, threshold = train_model(model, optimizer_ft, criterion_embeddings, criterion_classification, exp_lr_scheduler, keep_training_embeddings, num_epochs=num_epochs)
model_eval(model, threshold, test_loader)
print("train done")
torch.save(model, os.path.join(data_dir, args.load_model))
