from nlp.skipGram.utils import preprocess, create_lookup_tables, get_target, get_batches, cosine_similarity
from nlp.skipGram.SkipGram import SkipGram
from collections import Counter
from torch import nn
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# const
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim=300

# Read sample wiki data
with open('../../data/text8') as f:
    text = f.read()

words = preprocess(text)
vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words] # words are ordered with sentences.

# Subsampling
threshold = 1e-5
word_counts = Counter(int_words)

total_count = len(int_words)
freq = {word: occur/total_count for word, occur in word_counts.items()}
prop = {word: 1-threshold / freq[word] for word in word_counts}
train_words = [word for word in int_words if prop[word] <= random.random()]

# Train
model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 500
steps = 0
epochs = 5

for e in range(epochs):

    for inputs, targets in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
        inputs, targets = inputs.to(device), targets.to(device)

        log_ps = model(inputs)
        loss = criterion(log_ps, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
            _, closest_idx = valid_similarities.topk(6)


            valid_examples, closest_idx = valid_examples.to('cpu'), closest_idx.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idx[ii]][1:] # excluding self
                print(int_to_vocab[valid_idx.item()] + " | " + ", ".join(closest_words))
            print("...")

