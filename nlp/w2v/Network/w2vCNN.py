import torch.nn as nn
from nlp.w2v.globalconfig import *
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)  # 512 x 500 x 300, batch x sentence x dim
        self.conv1 = nn.Conv2d(1, 100, (3, 300))  # 512 x 3 x 500 x 300
        self.conv2 = nn.Conv2d(1, 100, (4, 300))
        self.conv3 = nn.Conv2d(1, 100, (5, 300))

        self.convs = nn.ModuleList([self.conv1, self.conv2, self.conv3])
        self.fc1 = nn.Linear(300, 1)
        self.fc2 = nn.Linear(250, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape(BATCH_SIZE, 1, 500, 300)  # batch x 1 x 500 x 300
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.convs]  # batch x 100 x 498 -> Filter grap one single word and neighbors to 1.(300)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in
             x]  # batch x 100 -> Pickup the max one from words in sentence
        x = torch.cat(x, 1)  # batch x 300 -> Connect all the filter with size 1
        x = self.dropout(x)
        x = self.sigmoid(self.fc1(x))
        return x
