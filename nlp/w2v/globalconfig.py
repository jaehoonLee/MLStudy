import torch

# Constant Value
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
MAX_VOCAB_SIZE = 25000
MIN_FREQ = 10
BATCH_SIZE = 512
MAX_SEQ_LEN = 500
W2V_DIM = 300
HIDDEN_DIM = 200