import torch
import torch.nn as nn

class ImageEmbedding(nn.Module):
    def __init__(self, convnet, num_classes, label_embeddings, fix_convnet=False):
        super(ImageEmbedding, self).__init__()
        self.model_ft = convnet
        self.num_ftrs = self.model_ft.fc.in_features
        print("num features {}, embedding dims {}".format(self.num_ftrs, label_embeddings.dim()))
        self.model_ft.fc = nn.Sequential()
        self.features_to_embedding1 = nn.Linear(self.num_ftrs, self.num_ftrs)
        self.features_to_embedding2 = nn.Linear(self.num_ftrs, label_embeddings.dim())
        self.normalization = torch.nn.BatchNorm1d(label_embeddings.dim(), track_running_stats=True)
        for param in convnet.parameters():
            param.requires_grad = not fix_convnet
        
    def forward(self, x):
        features = self.model_ft(x)
        features1 = self.features_to_embedding1(features)
        features2 = torch.tanh(features1)
        img_embedding = self.features_to_embedding2(features2)
        #img_embedding = self.normalization(img_embedding)
        img_embedding = img_embedding/torch.norm(img_embedding)
        return features, img_embedding
    
    def switch_convnet_learning(self, fix_convnet):
        for param in self.model_ft.parameters():
            param.requires_grad = not fix_convnet
            
class ImageEmbeddingClassifier(nn.Module):
    def __init__(self, imem_model, num_classes, label_embeddings, fix_convnet=False):
        super(ImageEmbeddingClassifier, self).__init__()
        self.imem_model = imem_model
        num_ftrs = imem_model.num_ftrs
        print("num features {}, embedding dims {}".format(num_ftrs, label_embeddings.dim()))
        self.classifier_fc = nn.Linear(num_ftrs+label_embeddings.dim(), num_classes)
        imem_model.switch_convnet_learning(fix_convnet)
        
    def forward(self, x):
        features, img_embedding = self.imem_model(x)
        features_and_embedding = torch.cat((features, img_embedding), 1)
        classification = self.classifier_fc(features_and_embedding)
        return classification, img_embedding
    
    def switch_convnet_learning(self, fix_convnet):
        self.imem_model.switch_convnet_learning(fix_convnet)
    
class ImageEmbeddingClassifierDistance(nn.Module):
    def __init__(self, imem_model, num_classes, label_embeddings, fix_convnet=False):
        super(ImageEmbeddingClassifierDistance, self).__init__()
        self.imem_model = imem_model
        num_ftrs = imem_model.num_ftrs
        print("num features {}, embedding dims {}".format(num_ftrs, label_embeddings.dim()))
        self.classifier_fc = nn.Linear(num_ftrs, num_classes)
        self.embedding_matrix = torch.tensor(label_embeddings.label_embeddings.wv.syn0norm).float().cuda().permute(0, 1)
        self.combine_classifiers = nn.Linear(2, 1)
        self.similarity = torch.nn.CosineSimilarity(dim=2)
        imem_model.switch_convnet_learning(fix_convnet)
        
    def forward(self, x):
        features, img_embedding = self.imem_model(x)
        classification = self.classifier_fc(features)
        distance = self.similarity(img_embedding.unsqueeze(1).repeat(1, 1103, 1), self.embedding_matrix.unsqueeze(0))
        self.combine_classifiers(torch.stack([classification, distance],dim=2))
        return classification, img_embedding
