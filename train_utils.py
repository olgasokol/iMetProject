import numpy as np
import torch
import torchvision
from torchvision import transforms
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    
def f2score(out, target, beta=2., ret_pr=False):
    eps = 0.0001
    #out - binary results
    tp = torch.sum(out * target).double()
    fp = torch.sum(out * (1-target)).double()
    fn = torch.sum((1-out)*target).double()
    tn = torch.sum((1-out)*(1-target)).double()
    p = tp / (tp+fp + eps)
    r = tp / (tp + fn + eps)
    score = (1+beta*beta)*p*r/(beta*beta*p + r + eps)
    acc = (tp+tn)/(tp+tn+fp+fn)
    if ret_pr:
        return score, p, r, acc
    return score

def find_threshold(outputs, target, step=0.1):
    best_thresholds = torch.zeros((1, outputs.shape[-1],), dtype=torch.float, device=device)
    best_f2 = 0
    
    for thr in np.arange(start=0, stop=1+step/2, step=step):
        pred = outputs > thr
        f2 = f2score(pred.int(), target)
        if f2 > best_f2:
            best_thresholds[0] = thr
            best_f2 = f2
            
    #best_f2  = torch.zeros(outputs.shape[-1], dtype=torch.double, device=device)

    for c in range(outputs.shape[-1]):
        for thr in np.arange(start=0, stop=1+step/2, step=step):
            thresholds = best_thresholds.clone()
            thresholds[:, c] = thr
            pred = outputs > thresholds
            f2 = f2score(pred.int(), target)
            if f2 > best_f2:
                best_thresholds = thresholds
                best_f2 = f2
            
    print(best_thresholds)
    #best_thresholds = 0.5
    return best_thresholds

def model_eval(model, threshold, test_loader):
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