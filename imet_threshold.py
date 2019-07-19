import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tutils
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
# from focal_loss_pytorch.focalloss import FocalLoss
from pytorch_workplace.focalloss import loss as FocalLoss
from skimage import io

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_data_transforms = transforms.Compose([
    torchvision.transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_data_transforms = transforms.Compose([
    torchvision.transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class IMetDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, labels_csv, annotations_csv, transform=None):
        super(IMetDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.labels_frame = pd.read_csv(labels_csv)
        self.annotations_frame = pd.read_csv(annotations_csv)

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
        labels = torch.LongTensor(labels)
        #labels = torch.DoubleTensor(labels)
        if self.transform:
            image = self.transform(image)
        return image, labels
    
    def getLabelWeights(self):
        label_idxs = self.annotations_frame.iloc[:, 1]
        label_idxs = label_idxs.values
        labels = [self.split_labels(l) for l in label_idxs]
        weights = np.sum(labels, axis=0)
        weights = np.ones(len(weights)) / weights
        weights = torch.DoubleTensor(weights)
        return weights.cuda()

    def split_labels(self, v):
        v=v.split(' ')
        v = np.asarray([np.int(l) for l in v])
        labels = np.zeros((len(self.labels_frame)))
        labels[v]=1
        return labels


# --------------------------------
# Hyper-parameters
# --------------------------------
dir_path = './data/train'
labels_path = './data/labels.csv'
annotations_path = './data/train.csv'
train_data = IMetDataset(root_dir=dir_path, labels_csv=labels_path, 
                         annotations_csv=annotations_path, transform=train_data_transforms)

num_epochs = 13
batch_size = 20
pretrained = True
learning_rate = 1e-4
dataset_size = len(train_data)
num_training = int(dataset_size * 0.6)
num_validation = int(dataset_size * 0.2)
dataset_size = {'train': num_training, 'val': num_validation}
fine_tune = True

num_classes = len(train_data.labels_frame)

val_data = IMetDataset(root_dir=dir_path, labels_csv=labels_path, 
                         annotations_csv=annotations_path,
                       transform=val_data_transforms)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
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


def train_model(model, optimizer, criterion,  scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch_i in range(num_epochs):
        print('Epoch {}/{}'.format(epoch_i, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            score_num = 0.

            # Iterate over data.
            for inputs, labels in loader[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).double()
                    loss = criterion(outputs.cpu(), labels.cpu())
                    preds = torch.nn.Softmax(outputs)
                    preds = preds.dim > threshold
                    preds = torch.IntTensor(preds.cpu().int())
                    score = f2sdcore(preds, torch.IntTensor(labels.cpu().int()))
                    score_num = score_num + 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if score_num % (5000/batch_size) == 0:
                   print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                       score_num, loss, score))
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += score

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / score_num

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def f2sdcore(out, target, beta=2.):
    eps = 0.0001
    #out - binary results
    tp = torch.sum(out * target).double()
    fp = torch.sum((out | target - target) * out).double()
    fn = torch.sum((out | target - target) * target).double()
    p = tp / (tp+fp + eps)
    r = tp / (tp + fn + eps)
    score = (1+beta*beta)*p*r/(beta*beta*p + r + eps)
    return score

def model_eval(model, treshold):
    model.eval()
    with torch.no_grad():
        score_total = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.nn.Softmax(outputs)
            preds = preds.dim > treshold
            preds = torch.IntTensor(preds.cpu().int())

            score = f2sdcore(preds, torch.IntTensor(labels.cpu().int()))
            total += 1
            score_total += score
    #         if total == 1000:
    #             break

        print('Accuracy of the network on the test images: {} %'.format(score_total / total))

print("model setup done")

# criterion =  nn.MultiLabelSoftMarginLoss()
# criterion =  nn.MultiLabelSoftMarginLoss(train_data.getLabelWeights())
gamma=2
criterion = FocalLoss.FocalLoss(gamma=gamma)

threshold = 0.9
# for i in range(0, 5):

#model_ft = ConvNet(num_classes, 524, pretrained)
model_ft = models.resnet18(pretrained=pretrained)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
print("model load done")

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.4)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
print("optimizer done")

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)
print("lr done")

# for param in model_ft.parameters():
#     param.requires_grad = False
# for param in model_ft.fc.parameters():
#     param.requires_grad = True

model_ft = model_ft.to(device)

print("training with threshold {}".format(threshold))
model_ft = train_model(model_ft, optimizer_ft, criterion, exp_lr_scheduler, num_epochs=num_epochs)
model_eval(model_ft,threshold )
print("train done")
torch.save(model_ft, './resnet18_tr__'+str(threshold)+"_focal_"+str(gamma))
