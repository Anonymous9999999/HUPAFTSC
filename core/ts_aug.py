import numpy as np

import torch
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def ts_aug(X,
           scale_ratio=1.1,
           jitter_ratio=0.8,
           permute_max_segments=5,
           is_sample_seq_first=False,
           to_torch=True):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    if is_sample_seq_first:
        X = np.transpose(X, (0, 2, 1))

    weak_aug = scaling(X, sigma=scale_ratio)
    strong_aug = jitter(permutation(X, max_segments=permute_max_segments),
                        sigma=jitter_ratio)

    if is_sample_seq_first:
        weak_aug = np.transpose(weak_aug, (0, 2, 1))
        strong_aug = np.transpose(strong_aug, (0, 2, 1))

    if to_torch:
        weak_aug = torch.Tensor(weak_aug)
        strong_aug = torch.Tensor(strong_aug)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return ret