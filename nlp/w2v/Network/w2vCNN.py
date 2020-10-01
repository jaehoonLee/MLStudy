import torch.nn as nn
from nlp.w2v.globalconfig import *
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)  # 512 x 500 x 300, batch x sentence x dim
        self.conv1 = nn.Conv2d(1, 16, 2, padding=1)  # 512 x 3 x 500 x 300
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 2, padding=1)
        self.fc1 = nn.Linear(32 * 125 * 75, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape(BATCH_SIZE, 1, 500, 300)
        x = self.pool(F.relu(self.conv1(x)))  # 512 x 3 x 250 x 150
        x = self.pool(F.relu(self.conv2(x)))  # 512 x 6 x 125 x 75
        x = x.reshape(BATCH_SIZE, -1)
        x = self.fc1(x)
        log_ps = self.sigmoid(x)

        return log_ps