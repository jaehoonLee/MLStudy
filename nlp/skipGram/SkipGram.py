from torch import nn


class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)

        return log_ps
