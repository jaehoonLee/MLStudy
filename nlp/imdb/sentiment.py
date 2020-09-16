import torch
import random
from torchtext import data, datasets

SEED = 1234

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT = data.Field(pad_first=True, fix_length=500)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(test_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(LABEL.vocab.freqs.most_common(20))

BATCH_SIZE = 64
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)
