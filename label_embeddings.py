from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

class LabelEmbeddings:
    def __init__(self, labels_path, annotations_path, size=10, load_path=""):  
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
            self.label_embeddings = Word2Vec(min_count=1, size=size)
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

