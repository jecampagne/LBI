import jax
import jax.numpy as np
import numpy as onp
import torch


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset class for all datasets.
    """

    def __init__(self, X, Theta, **kwargs):
        super(BaseDataset, self).__init__()
        assert (
            X.shape[0] == Theta.shape[0]
        ), "X and Theta must have the same number of rows"
        self.X = X
        self.Theta = Theta

    def __getitem__(self, index):
        x = self.X[index]
        theta = self.Theta[index]
        return x, theta

    def __len__(self):
        return self.X.shape[0]


class GaussianNoiseDataset(BaseDataset):
    """
    Add gaussian noise on demand

    logged_X_idx: indices of X that are logged. Adding noise to these
                    requires exponentiating the data first


    NOTE: The gaussian noise is added to the transformed data,
            NOT on the original data.
            Make sure to transform sigma accordingly
    """

    def __init__(
        self, X, Theta, sigma=None, scale_X=None, inverse_scale_X=None, **kwargs
    ):
        super(GaussianNoiseDataset, self).__init__(X, Theta, **kwargs)

        if scale_X is None:
            scale_X = lambda x: x
        if inverse_scale_X is None:
            inverse_scale_X = lambda x: x
        
        assert sigma is not None, "sigma must be specified"
        
        self.X = X
        self.Theta = Theta
        self.sigma = torch.tensor(onp.array(sigma))
        self.scale_X = scale_X
        self.inverse_scale_X = inverse_scale_X

    def __getitem__(self, index):
        x = self.X[index]
        theta = self.Theta[index]

        # transform to data space, add gaussian noise, transform back
        x = self.inverse_scale_X(x)
        x = x + self.sigma * torch.randn_like(x)
        x = self.scale_X(x)

        return x, theta
