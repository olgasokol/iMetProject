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
from image_embedding_models import *
import argparse

parser = argparse.ArgumentParser(description='Train iMet classifier')
parser.add_argument('--data_dir', type=str, default='/raid/hlcv19/team22/imet/data/', help='iMet dataset location')
parser.add_argument('--labels_file', type=str, help='Labels file name', default='labels.csv')
parser.add_argument('--train_file', type=str, help='Train file name', default='train_.csv')
parser.add_argument("--convnet_model", type=str, default='/raid/hlcv19/team22/imet/data/resnet18_tr__0.9_focal_gamma2')
parser.add_argument("--fix_convnet", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--embedding_dim", type=int, default=10)
parser.add_argument("--load_model", type=str)
parser.add_argument("--save_model", type=str, default='/raid/hlcv19/team22/imet/data/resnet18_embeddings_trainable_resnet_size_10')
parser.add_argument("--label2vec_model", type=str, default='/raid/hlcv19/team22/imet/data/label2v_10.model')

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

label_embeddings = LabelEmbeddings(labels_path, annotations_path, size=args.embedding_dim, load_path=args.label2vec_model)
print("labels done")

train_data = IMetDataset(root_dir=dir_path, labels_csv=labels_path,
                         annotations_csv=annotations_path, 
                         transform=train_data_transforms)


num_epochs = args.epochs
batch_size = args.batch_size
embedding_dim = label_embeddings.dim()
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

def train_model(model, optimizer, criterion_embeddings,  scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999.0
    losses_train = []
    losses_val = []

    for epoch_i in range(num_epochs):
        print('Epoch {}/{}'.format(epoch_i, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss_embedding = 0.0
            running_corrects = 0
            score_num = 0.
            
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
                    loss_embedding = criterion_embeddings(output_embeddings.view(-1, embedding_dim), embeddings.view(-1, embedding_dim), y.view(-1))
                    score_num = score_num + 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_embedding.backward()
                        optimizer.step()
                if score_num % (5000/batch_size) == 0:
                    print('{} Loss embedding: {:.4f}'.format(
                        score_num, loss_embedding))
                # statistics
                running_loss_embedding += loss_embedding.item() * inputs.size(0)

            epoch_loss_embedding = running_loss_embedding / dataset_size[phase]
            print('{} Loss embedding: {:.4f}'.format(
                phase, epoch_loss_embedding))
            losses_train.append(epoch_loss_embedding)

            # deep copy the model
            if phase == 'val':
                losses_val.append(epoch_loss_embedding)
                if epoch_loss_embedding < best_loss:
                    best_loss = epoch_loss_embedding
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, args.save_model)
                    print('model saved')

    print("losses train {}".format(losses_train))
    print("losses validation {}".format(losses_val))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(history)em
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

criterion_embeddings = nn.CosineEmbeddingLoss()

if args.load_model:
    model = torch.load(os.path.join(args.load_model))
else:
    resnet = torch.load(args.convnet_model)
    model = ImageEmbedding(resnet, num_classes, label_embeddings, fix_convnet=False)
            
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
model = train_model(model, optimizer_ft, criterion_embeddings, exp_lr_scheduler, num_epochs=num_epochs)
print("train done")
torch.save(model, args.save_model)
