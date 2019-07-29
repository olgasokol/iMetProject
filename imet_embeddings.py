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
import random
import pandas as pd
# from focal_loss_pytorch.focalloss import FocalLoss
from pytorch_workplace.focalloss import loss as FocalLoss
from skimage import io
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

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
    def __init__(self, root_dir, labels_csv, annotations_csv, label_embeddings, transform=None):
        super(IMetDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.labels_frame = pd.read_csv(labels_csv)
        self.annotations_frame = pd.read_csv(annotations_csv)
        self.label_embeddings = label_embeddings
        
        labels = [[int(x) for x in s.split(' ')] for s in self.annotations_frame.iloc[:, 1]]
        flat_labels = []
        for l in labels:
              flat_labels.extend(l)
        self.label_weights = 1/np.unique(flat_labels, return_counts=True)[1]
        
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
        labels = torch.DoubleTensor(labels)
        if self.transform:
            image = self.transform(image)
        return image, labels
    
    def getLabelWeights(self):
        print("getLabelWeights")
        label_idxs = self.annotations_frame.iloc[:, 1]
        label_idxs = label_idxs.values
        labels = [self.split_labels(l) for l in label_idxs]
        weights = np.sum(labels, axis=0)
        print(weights)
        weights = np.ones(len(weights)) / weights
        weights = torch.tensor(weights, device="cuda", dtype=torch.double)
        return weights

    def split_labels(self, v):
        v=v.split(' ')
        v = np.asarray([np.int(l) for l in v])
        labels = np.zeros((len(self.labels_frame)))
        labels[v]=1
        return labels
    
class LabelEmbeddings:
    def __init__(self, labels_path, annotations_path, load_path=""):  
        self.labels_to_ix = {}
        self.ix_to_label = {}

        with open(labels_path) as labels_f:
            lines = labels_f.readlines()
            for l in lines[1:]:
                idx, label = l.strip().split(',')
                self.labels_to_ix[label] = int(idx)
                self.ix_to_label[int(idx)] = label

        self.vocab = list(self.labels_to_ix.keys())
        vocab_size = len(self.vocab)

        words_context = [] 
        with open(annotations_path) as labels_f:
            lines = labels_f.readlines()
            for line in lines[1:]:
                labels = line.strip().split(',')[1].split()
                labels = [int(l) for l in labels]
                for i in range(len(labels)):
                    for j in range(i+1, len(labels)):
                        words_context.append((self.ix_to_label[labels[i]], self.ix_to_label[labels[j]]))

        if load_path == "":
            self.label_embeddings = Word2Vec(min_count=1)
            self.label_embeddings.build_vocab(words_context)  # prepare the model vocabulary
            self.label_embeddings.train(words_context, total_examples=self.label_embeddings.corpus_count, 
                                        epochs=self.label_embeddings.iter)  # train word vectors
        else:
            self.label_embeddings = Word2Vec.load(load_path)
        self.label_embeddings.init_sims()
    
    def save(self, path):
        self.label_embeddings.save(path)
    
    def __getitem__(self, idx):
        return self.label_embeddings.wv.word_vec(self.ix_to_label[idx], use_norm=True)
    
    def dim(self):
        return len(self.label_embeddings.wv[self.vocab[0]])
    
    def most_similar(self, label):
        return self.label_embeddings.most_similar(positive=label)


# --------------------------------
# Hyper-parameters
# --------------------------------
data_dir = '/raid/hlcv19/team22/imet/data/'
dir_path = os.path.join(data_dir, 'train')
labels_path = os.path.join(data_dir, 'labels.csv')
annotations_path = os.path.join(data_dir, 'train_.csv')

label_embeddings = LabelEmbeddings(labels_path, annotations_path, load_path=os.path.join(data_dir, 'label2v.model'))
print("labels done")

train_data = IMetDataset(root_dir=dir_path, labels_csv=labels_path,
                         annotations_csv=annotations_path,  label_embeddings = label_embeddings, 
                         transform=train_data_transforms)


num_epochs = 40
batch_size = 20
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
                         annotations_csv=annotations_path, label_embeddings = label_embeddings,
                       transform=val_data_transforms)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

class ImageEmbeddingClassifier(nn.Module):
    def __init__(self, convnet, num_classes):
        super(ImageEmbeddingClassifier, self).__init__()
        self.model_ft = convnet
        num_ftrs = self.model_ft.fc.in_features
        print("num features {}, embedding dims {}".format(num_ftrs, label_embeddings.dim()))
        self.model_ft.fc = nn.Sequential()
        self.features_to_embedding1 = nn.Linear(num_ftrs, num_ftrs)
        self.features_to_embedding2 = nn.Linear(num_ftrs, label_embeddings.dim())
        self.normalization = torch.nn.BatchNorm1d(label_embeddings.dim(), track_running_stats=True)
        self.classifier_fc = nn.Linear(num_ftrs+label_embeddings.dim(), num_classes)
        for param in convnet.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        features = self.model_ft(x)
        features = self.features_to_embedding1(features)
        features = torch.tanh(features)
        img_embedding = self.features_to_embedding2(features)
        #img_embedding = self.normalization(img_embedding)
        img_embedding = img_embedding/torch.norm(img_embedding)
        features_and_embedding = torch.cat((features, img_embedding), 1)
        classification = self.classifier_fc(features_and_embedding)
        return classification, img_embedding
    
    def get_embedding_params(self):
        return self.features_to_embedding.parameters()
    
class ImageEmbeddingClassifierDistance(nn.Module):
    def __init__(self, convnet, num_classes):
        super(ImageEmbeddingClassifierDistance, self).__init__()
        self.model_ft = convnet
        num_ftrs = self.model_ft.fc.in_features
        print("num features {}, embedding dims {}".format(num_ftrs, label_embeddings.dim()))
        self.model_ft.fc = nn.Sequential()
        self.features_to_embedding = nn.Linear(num_ftrs, label_embeddings.dim())
        self.classifier_fc = nn.Linear(num_ftrs, num_classes)
        self.embedding_matrix = torch.tensor(label_embeddings.label_embeddings.wv.syn0).float().cuda().permute(0, 1)
        self.combine_classifiers = nn.Linear(2, 1)
        self.similarity = torch.nn.CosineSimilarity(dim=2)
        
    def forward(self, x):
        features = self.model_ft(x)
        img_embedding = self.features_to_embedding(features)
        classification = self.classifier_fc(features)
        distance = self.similarity(img_embedding.unsqueeze(1).repeat(1, 1103, 1), self.embedding_matrix.unsqueeze(0))
        self.combine_classifiers(torch.stack([classification, distance],dim=2))
        return classification, img_embedding


def onehot2vectors(onehot, label_embeddings, padded_len=15):
    embedding_dim = label_embeddings.dim()
    res = np.zeros((onehot.shape[0], padded_len, embedding_dim,))
    y = np.ones((onehot.shape[0], padded_len))
    for i in range(onehot.shape[0]):
        labels = torch.nonzero(onehot[i]).flatten().cpu().detach().tolist()
        j=0
        for l in labels:
            res[i][j] = label_embeddings[l]
            j+=1
        while j < padded_len:
            l = random.randint(0,1102)
            while l in labels:
                l = random.randint(0,1102)
            res[i][j] = label_embeddings[l]
            y[i, j]=-1
            j+=1
            
    return torch.tensor(res, dtype=torch.double, device=device, requires_grad=False), torch.tensor(y, dtype=torch.double, device=device, requires_grad=False)
    
    
def train_model(model, optimizer, criterion_embeddings, criterion_classification,  scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999.0

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
            running_loss_classification = 0.0
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
                    #print(output_embeddings.shape, embeddings.shape)
                    loss_embedding = criterion_embeddings(output_embeddings.view(-1, 100), embeddings.view(-1, 100), y.view(-1))
                    
                    #loss_embedding = criterion_embeddings(output_embedding.double(), embeddings)
                    loss_classifier = criterion_classification(output_classes.double(), labels)
                    preds = torch.sigmoid(output_classes)
                    preds = preds > threshold
                    #preds = preds.int()
                    score = f2score(preds.int(), labels.int())
                    score_num = score_num + 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss_embedding#+loss_classifier
                        loss.backward()
                        optimizer.step()
                if score_num % (5000/batch_size) == 0:
                    print('{} Loss embedding: {:.4f}, Loss classification: {:.4f}, Acc: {:.4f}'.format(
                        score_num, loss_embedding, loss_classifier, score))
                # statistics
                running_loss_embedding += loss_embedding.item() * inputs.size(0)
                running_loss_classification += loss_classifier.item() * inputs.size(0)
                running_corrects += score

            epoch_loss_embedding = running_loss_embedding / dataset_size[phase]
            epoch_loss_classification = running_loss_classification / dataset_size[phase]
            epoch_acc = running_corrects / score_num

            print('{} Loss embedding: {:.4f}, Loss classification: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch_loss_embedding, epoch_loss_classification, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss_embedding < best_loss:#epoch_acc > best_acc:
                best_loss = epoch_loss_embedding
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join(data_dir, './resnet18_embeddings_with_classification_epoch_{}_loss_{:.4f}'.format(epoch_i, best_loss)))
                print('model saved')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def f2score(out, target, beta=2.):
    eps = 0.0001
    #out - binary results
    tp = torch.sum(out * target).double()
    fp = torch.sum(out * (1-target)).double()
    fn = torch.sum((1-out)*target).double()
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
            outputs, _ = model(images)
            preds = torch.sigmoid(outputs)
            preds = preds > threshold

            score = f2score(out=preds.int(), target=labels.int())
            total += 1
            score_total += score
    #         if total == 1000:
    #             break

        print('Accuracy of the network on the test images: {} %'.format(score_total / total))

print("model setup done")

weighted_classes = False
print("using weighted classification {}".format(weighted_classes))

criterion_embeddings = nn.CosineEmbeddingLoss()
#criterion_embeddings = nn.MSELoss()
criterion_classification =  nn.BCEWithLogitsLoss()
if weighted_classes:
    criterion_classification =  nn.BCEWithLogitsLoss(train_data.getLabelWeights())
gamma=2
#criterion = FocalLoss.FocalLoss(gamma=gamma)

threshold = 0.5
# for i in range(0, 5):

#model_ft = ConvNet(num_classes, 524, pretrained)
#model_ft = torch.load(os.path.join(data_dir, 'resnet18_tr__0.9_focal_gamma2'))
#model = ImageEmbeddingClassifier(model_ft, num_classes)
model = torch.load(os.path.join(data_dir, 'resnet18_embeddings'))
print("model load done")


# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.4)
#optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
optimizer_ft = optim.Adam(model.parameters())
print("optimizer done")

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print("lr done")

# for param in model_ft.parameters():
#     param.requires_grad = False
# for param in model_ft.fc.parameters():
#     param.requires_grad = True

#model_ft = model_ft.to(device)
model = model.to(device)

print("training with threshold {}".format(threshold))
model_ft = train_model(model, optimizer_ft, criterion_embeddings, criterion_classification, exp_lr_scheduler, num_epochs=num_epochs)
model_eval(model,threshold )
print("train done")
torch.save(model, os.path.join(data_dir, './resnet18_embeddings_with_classification'))
