import torch.nn as nn
import torchtext
import torch.optim as optim
import time
from torchtext.data import get_tokenizer
from torchtext.experimental.datasets import IMDB
from torch.utils.data import random_split, DataLoader
from nlp.w2v.Network.w2vDense import Sentiment
from nlp.w2v.utils import *

# Prepare Data
vocab = IMDB(data_select='train')[0].get_vocab()
glove_vocab = torchtext.vocab.Vocab(counter=vocab.freqs,
                                    max_size=MAX_VOCAB_SIZE,
                                    min_freq=MIN_FREQ,
                                    vectors=torchtext.vocab.GloVe(name='6B', dim=W2V_DIM))

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
train_set, test_set = IMDB(tokenizer=tokenizer, vocab=glove_vocab)
vocab = train_set.get_vocab()

generator = torch.Generator().manual_seed(SEED)
train_num = int(len(train_set) * 0.8)
valid_num = len(train_set) - train_num
train_set, valid_set = random_split(train_set, lengths=[train_num, valid_num], generator=generator)
pad_id = vocab['<pad>']

# Zero Padding
def pad_trim(data):
    ''' Pads or trims the batch of input data.

    Arguments:
        data (torch.Tensor): input batch
    Returns:
        new_input (torch.Tensor): padded/trimmed input
        labels (torch.Tensor): batch of output target labels
    '''
    data = list(zip(*data))
    # Extract target output labels
    labels = torch.tensor(data[0]).float().to(DEVICE)
    # Extract input data
    inputs = data[1]

    # Extract only the part of the input up to the MAX_SEQ_LEN point
    # if input sample contains more than MAX_SEQ_LEN. If not then
    # select entire sample and append <pad_id> until the length of the
    # sequence is MAX_SEQ_LEN
    new_input = torch.stack([torch.cat((input[:MAX_SEQ_LEN],
                                        torch.tensor([pad_id] * max(0, MAX_SEQ_LEN - len(input))).long()))
                             for input in inputs])

    return new_input, labels

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=pad_trim)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, collate_fn=pad_trim)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=pad_trim)

# Define Model
model = Sentiment(vocab).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training
epochs = 10
for e in range(epochs):
    running_loss = 0
    start = time.time()
    for input, label in train_loader:
        if input.shape[0] != BATCH_SIZE:
            continue

        input, label = input.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        output = model(input).squeeze(1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        test_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for input, labels in valid_loader:
                if input.shape[0] != BATCH_SIZE:
                    continue
                input, labels = input.to(DEVICE), labels.to(DEVICE)
                output = model(input).squeeze(1)

                test_loss += criterion(output, labels)
                eval_acc += calculate_acc(output, labels)

    print("==============================================")
    print(f"Device = {DEVICE}; Elapsed time {(time.time() - start):.3f} seconds")
    print(f'Training loss: {running_loss / len(train_loader)}')
    print(f'Validation loss: {test_loss / len(valid_loader)}')
    print(f'Correct Rate: {eval_acc / len(valid_loader)}')

# Test
test_loss = 0
eval_acc = 0
start = time.time()
with torch.no_grad():
    for input, labels in test_loader:
        if input.shape[0] != BATCH_SIZE:
            continue
        input, labels = input.to(DEVICE), labels.to(DEVICE)
        output = model(input).squeeze(1)
        test_loss += criterion(output, labels)
        eval_acc += calculate_acc(output, labels)
print("==============================================")
print(f"Device = {DEVICE}; Elapsed time {(time.time() - start):.3f} seconds")
print(f'Test loss: {test_loss / len(test_loader)}')
print(f'Correct Rate: {eval_acc / len(test_loader)}')