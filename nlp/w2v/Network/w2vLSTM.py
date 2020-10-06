import torch.nn as nn
from nlp.w2v.globalconfig import *
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)  # batch x 500 x 300, batch x sentence x dim
        self.lstm = nn.LSTM(W2V_DIM, HIDDEN_DIM, 2, dropout=0.5, batch_first=True)

        self.fc1 = nn.Linear(HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.embed(x)  # batch x 500 x 300
        output, (hidden, cell) = self.lstm(x)  # batch x 500 x 100

        # Pick up last one
        #         x = output[:,-1] # Extract Last
        #         x = self.dropout(x)
        #         x = self.sigmoid(self.fc1(x))

        # Stack up and pick up last one
        x = output.contiguous().view(-1, HIDDEN_DIM)
        x = self.dropout(x)
        x = self.sigmoid(self.fc1(x))
        x = x.view(BATCH_SIZE, -1)
        x = x[:, -1]
        return x