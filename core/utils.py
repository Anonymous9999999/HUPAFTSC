import os
import json
import torch
import ipdb
import numpy as np
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters


def compute_l2_norm_for_model(model):
    with torch.no_grad():
        params_l2_list = []
        for name, params in model.named_parameters():
            if params is not None:
                p_norm = params.data.norm(2).item()
                params_l2_list.append(p_norm)
    return np.array(params_l2_list)


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def check_nan(to_check_data, model):
    try:
        detect_nan_parameters(model)
    except:
        ipdb.set_trace()

    for data in to_check_data:
        if bool(torch.isnan(data).any()):
            ipdb.set_trace()


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def get_x_true_len(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()

    if len(np.where(x == 0.0)[0]) == 0:
        return len(x)
    else:
        x_reverse = x[::-1, :]
        valid_len = None
        for i, x_ in enumerate(x_reverse):
            if np.sum(x_) != 0.0:
                valid_len = len(x_reverse) - i
                break
        assert valid_len is not None
        return valid_len


def get_X_padding_masks(X):
    """
    Parameters
    ----------
    X

    Returns
    -------

    """
    if isinstance(X, np.ndarray):
        X = torch.Tensor(X)
    if isinstance(X, list):
        X = torch.stack(X).float()  # (batch_size, seq_len, feat_dimension)

    lengths = []
    for x in X:
        x_length = get_x_true_len(x)
        lengths.append(x_length)

    max_len = X.shape[1]
    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16),
        max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    assert X.shape[:2] == padding_masks.shape, ipdb.set_trace()
    return padding_masks, lengths
