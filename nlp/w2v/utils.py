import torch
from nlp.w2v.globalconfig import *

def calculate_acc(output, target):
    ''' Calculates binary accuracy based
        on given predictions and target labels.

    Arguments:
        output (torch.tensor): predictions
        target (torch.tensor): target labels
    Returns:
        acc (float): binary accuracy
    '''
    output = torch.round(output)
    correct = torch.sum(output == target).float()
    acc = (correct / len(target)).item()
    return acc