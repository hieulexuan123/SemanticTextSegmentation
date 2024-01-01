import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import json
from sentence_transformers import SentenceTransformer


class ArticleDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, file_path):
        data = []
        model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.loads(f.read())
            for article in articles:
                labels = [1 if label == 'b' else 0 for label in article['labels']]
                embeddings = model.encode(article['sentences'])
                assert len(embeddings) == len(labels)
                data.append((embeddings, labels))
        return data


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    # Pad the embeddings
    padded_embeddings = pad_sequence([torch.tensor(e) for e in embeddings], batch_first=True, padding_value=0.0)
    # Pad the labels
    padded_labels = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=0)
    # Create the masks
    masks = torch.zeros_like(padded_labels, dtype=torch.bool)
    for i, length in enumerate([len(l) for l in labels]):
        masks[i, :length] = 1  # Mark the actual label positions as True
    return padded_embeddings, padded_labels, masks


def get_data_loader(mode, batch_size, shuffle):
    switcher = {
        "train": ("train_set.pth", "train.txt"),
        "val": ("val_set.pth", "val.txt"),
        "test": ("test_set.pth", "test.txt")
    }
    if os.path.exists(switcher[mode][0]):
        article_dataset = torch.load(switcher[mode][0])
    else:
        article_dataset = ArticleDataset(os.path.join('dataset', switcher[mode][1]))
        torch.save(article_dataset, switcher[mode][0])
    return DataLoader(article_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
