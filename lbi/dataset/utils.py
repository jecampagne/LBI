import torch
import numpy as onp
from .base import BaseDataset, GaussianNoiseDataset
from .collate_fns import FlowCollate_fn, LikelihoodRatioCollate_fn
from sklearn.model_selection import train_test_split


def remove_nans(*arrs):
    """
    Remove nans from all arrays while preserving indices
    """
    non_nan_idx = onp.ones(arrs[0].shape[0])
    
    for arr in arrs:
        non_nan_idx = onp.logical_and(non_nan_idx, ~onp.isnan(arr).any(axis=1))
    
    output_arrs = []
    for arr in arrs:
        output_arrs.append(arr[non_nan_idx])
    return output_arrs

    
    

def getDataLoaderBuilder(
    sequential_mode,
    batch_size,
    train_split=0.9,
    num_workers=0,
    add_noise=True,
    **kwargs
):
    """
    Get data loaders for a dataset.
    """
    assert sequential_mode in [
        "flow",
        "classifier",
    ], "sequential_mode must be 'flow' or 'classifier'"

    if sequential_mode == "flow":
        collate_fn = None
    elif sequential_mode == "classifier":
        collate_fn = LikelihoodRatioCollate_fn

    if add_noise:
        DSet = GaussianNoiseDataset
    else:
        DSet = BaseDataset

    def data_loader_builder(X, Theta):

        # remove nans from data (if any)
        X, Theta = remove_nans(X, Theta)

        train_X, valid_X, train_Theta, valid_Theta = train_test_split(
            X, Theta, train_size=train_split
        )

        train_X = torch.tensor(onp.array(train_X), dtype=torch.float32)
        valid_X = torch.tensor(onp.array(valid_X), dtype=torch.float32)
        train_Theta = torch.tensor(onp.array(train_Theta), dtype=torch.float32)
        valid_Theta = torch.tensor(onp.array(valid_Theta), dtype=torch.float32)

        trainDataset = DSet(
            train_X,
            train_Theta,
            **kwargs
        )
        validDataset = DSet(
            valid_X,
            valid_Theta,
            **kwargs
        )

        trainLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        validLoader = torch.utils.data.DataLoader(
            validDataset,
            batch_size=batch_size * 10,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return trainLoader, validLoader

    return data_loader_builder
