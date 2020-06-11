import torch
import numpy as np
from models import *
from utils import *

def detect_MRs(model, train_loader, test_loader, device, mr_types=None):

    supported_mrs = [
                      # 'translation', 
                      # 'order_permutation',
                      # 'feature_permutation',
                      'rotation'
                    ]

    if mr_types is None:
        mr_types = supported_mrs
    
    for mr in mr_types:
        if mr not in supported_mrs:
            raise ValueError("Requested MR: '" + mr + "' is not currently supported. \n \
                Please select one or more of the following: " + str(supported_mrs))

    X_train, y_train = train_loader.dataset.tensors
    X_test, y_test = test_loader.dataset.tensors

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model.to(device)

    X_train_dims = X_train.shape
    num_X_train = X_train_dims[0]

    X_test_dims = X_test.shape
    num_X_test = X_test_dims[0]

    classes = y_train.unique()
    num_y = len(classes)

    out = {}

    if 'rotation' in mr_types and len(X_test_dims) > 2:
        similarity_threshold = 0.50
        class_mrs = {}
        for k in [1,2,3]:
            for i in classes:
                idx_i = y_test == i
                for j in classes:
                    idx_j = y_test == j
                    X_test_ = X_test.clone()
                    X_test_j = torch.unsqueeze(torch.rot90(torch.squeeze(X_test_[idx_j]),k=k,dims=(1,2)), 1)
                    y_test_i = torch.ones((len(X_test_j))).to(device) * i
                    ij_acc = get_accuracy(model, X_test_j, y_test_i)
                    if ij_acc > similarity_threshold:
                        ii = i.item()
                        jj = j.item()
                        ij_acc = round(ij_acc.item(),2)
                        if i == j:
                            class_mrs[(k * 90, ii, jj, ij_acc)] = 'invariance'
                        elif i != j:
                            class_mrs[(k * 90, ii, jj, ij_acc)] = 'sibylvariance'
        out['rotation'] = class_mrs

    return out