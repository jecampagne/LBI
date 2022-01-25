import jax
import jax.numpy as np


def get_scaler_kwargs(scaler_type, data):
    if scaler_type.lower() not in ["identity", "standard", "minmax"]:
        raise ValueError("Unknown scaler name: {}".format(scaler_type)) 
    
    d = {"scaler_type": scaler_type.lower()}
    d["mean"] = np.mean(data, axis=0)
    d["std"] = np.std(data, axis=0)
    d["min"] = np.min(data, axis=0)
    d["max"] = np.max(data, axis=0)
    return d


def get_scaler(**kwargs):
    def IdentityScaler(data, forward=True):
        if forward:
            return data
        else:
            return data

    def StandardScaler(data, forward=True):
        if "mean" not in kwargs:
            raise ValueError("mean must be provided")
        if "std" not in kwargs:
            raise ValueError("std must be provided")

        if forward:
            return (data - kwargs["mean"]) / kwargs["std"]
        else:
            return data * kwargs["std"] + kwargs["mean"]

    def MinMaxScaler(data, forward=True):
        if "max" not in kwargs:
            raise ValueError("max must be provided")
        if "min" not in kwargs:
            raise ValueError("min must be provided")

        if forward:
            return (data - kwargs["min"]) / (kwargs["max"] - kwargs["min"])
        else:
            return data * (kwargs["max"] - kwargs["min"]) + kwargs["min"]

    if kwargs['scaler_type'].lower() == "identity":
        return IdentityScaler
    elif kwargs['scaler_type'].lower() == "standard":
        return StandardScaler
    elif kwargs['scaler_type'].lower() == "minmax":
        return MinMaxScaler
    else:
        raise ValueError("Unknown scaler name: {}".format(kwargs['scaler_type']))
