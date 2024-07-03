import matplotlib.pyplot as plt
import torch
from torch.autograd import grad, Variable
import autograd
import copy
import scipy as sp
from scipy import stats
from sklearn import metrics
import sys
import ot
import gwot
from gwot import models, sim, ts, util
import gwot.bridgesampling as bs
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
print(torch.cuda.is_available())

if __name__ == "__main__":
    print('hi')