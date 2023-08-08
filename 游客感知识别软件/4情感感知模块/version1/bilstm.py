import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

nltk.download('punkt')

import pandas as pd
import numpy as np

data = pd.read_csv(r'twitter_training.csv', header=None, names=['Id', 'Pc', 'Se', 'Co'])
data.to_csv('twitter_training_new.csv', index=False)

data = pd.read_csv(r'twitter_validation.csv', header=None, names=['Id', 'Pc', 'Se', 'Co'])
data.to_csv('twitter_validation_new.csv', index=False)

train_path = "twitter_training_new.csv"
test_path = "twitter_validation_new.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = train_df.drop(columns=["Id"])
train_df = train_df.drop(columns=["Pc"])
train_df = train_df.dropna()
train_df = train_df[train_df['Se'] != "Irrelevant"]
train_df = train_df[train_df['Co'] != "Not Available"]

test_df = test_df.drop(columns=["Id"])
test_df = test_df.drop(columns=["Pc"])
test_df = test_df.dropna()
test_df = test_df[test_df['Se'] != "Irrelevant"]
test_df = test_df[test_df['Co'] != "Not Available"]

train_df['Se'].value_counts()
train_clean_df, test_clean_df = train_test_split(train_df, test_size=0.15)

train_set = list(train_clean_df.to_records(index=False))
test_set = list(test_clean_df.to_records(index=False))


def remove_links_mentions(tweet):
    link_re_pattern = "https?:\/\/t.co/[\w]+"
    mention_re_pattern = "@\w+"
    tweet = re.sub(link_re_pattern, "", tweet)
    tweet = re.sub(mention_re_pattern, "", tweet)
    return tweet.lower()


train_set = [(label, word_tokenize(remove_links_mentions(tweet))) for label, tweet in train_set]
test_set = [(label, word_tokenize(remove_links_mentions(tweet))) for label, tweet in test_set]

index2word = ["<PAD>", "<SOS>", "<EOS>"]

for ds in [train_set, test_set]:
    for label, tweet in ds:
        for token in tweet:
            if token not in index2word:
                index2word.append(token)

word2index = {token: idx for idx, token in enumerate(index2word)}


def label_map(label):
    if label == "Negative":
        return 0
    elif label == "Neutral":
        return 1
    else:  # Positive
        return 2


seq_length = 64


def encode_and_pad(tweet, length):
    sos = [word2index["<SOS>"]]
    eos = [word2index["<EOS>"]]
    pad = [word2index["<PAD>"]]

    if len(tweet) < length - 2:  # -2 for SOS and EOS
        n_pads = length - 2 - len(tweet)
        encoded = [word2index[w] for w in tweet]
        return sos + encoded + eos + pad * n_pads
    else:  # tweet is longer than possible; truncating
        encoded = [word2index[w] for w in tweet]
        truncated = encoded[:length - 2]
        return sos + truncated + eos


train_encoded = [(encode_and_pad(tweet, seq_length), label_map(label)) for label, tweet in train_set]
test_encoded = [(encode_and_pad(tweet, seq_length), label_map(label)) for label, tweet in test_set]

batch_size = 50

train_x = np.array([tweet for tweet, label in train_encoded])
train_y = np.array([label for tweet, label in train_encoded])
test_x = np.array([tweet for tweet, label in test_encoded])
test_y = np.array([label for tweet, label in test_encoded])

train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)


class BiLSTM_SentimentAnalysis(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embs, hidden)

        # Dropout is applied to the output and fed to the FC layer
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]
        return out, hidden

    def init_hidden(self):
        return (torch.zeros(1, batch_size, 32), torch.zeros(1, batch_size, 32))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM_SentimentAnalysis(len(word2index), 64, 32, 0.2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 100
losses = []
for e in range(epochs):

    h0, c0 = model.init_hidden()

    h0 = h0.to(device)
    c0 = c0.to(device)

    for batch_idx, batch in enumerate(train_dl):
        input = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out, hidden = model(input, (h0, c0))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
    losses.append(loss.item())

plt.plot(losses)

batch_acc = []
for batch_idx, batch in enumerate(test_dl):
    input = batch[0].to(device)
    target = batch[1].to(device)

    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        out, hidden = model(input, (h0, c0))
        _, preds = torch.max(out, 1)
        preds = preds.to("cpu").tolist()
        batch_acc.append(accuracy_score(preds, target.tolist()))

sum(batch_acc) / len(batch_acc)
