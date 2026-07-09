# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utility functions for global alignment
# --------------------------------------------------------
"""Module for base_models -> three_dimensions -> general_3d -> stable_virtual_camera -> stable_virtual_camera_runtime -> third_party -> dust3r -> dust3r -> cloud_opt -> commons.py functionality."""

import numpy as np
import torch
import torch.nn as nn


def edge_str(i, j):
    """Edge str.

    Args:
        i: The i.
        j: The j.
    """
    return f"{i}_{j}"


def i_j_ij(ij):
    """I j ij.

    Args:
        ij: The ij.
    """
    return edge_str(*ij), ij


def edge_conf(conf_i, conf_j, edge):
    """Edge conf.

    Args:
        conf_i: The conf i.
        conf_j: The conf j.
        edge: The edge.
    """
    return float(conf_i[edge].mean() * conf_j[edge].mean())


def compute_edge_scores(edges, conf_i, conf_j):
    """Compute edge scores.

    Args:
        edges: The edges.
        conf_i: The conf i.
        conf_j: The conf j.
    """
    return {(i, j): edge_conf(conf_i, conf_j, e) for e, (i, j) in edges}


def NoGradParamDict(x):
    """Nogradparamdict.

    Args:
        x: The x.
    """
    assert isinstance(x, dict)
    return nn.ParameterDict(x).requires_grad_(False)


def get_imshapes(edges, pred_i, pred_j):
    """Get imshapes.

    Args:
        edges: The edges.
        pred_i: The pred i.
        pred_j: The pred j.
    """
    n_imgs = max(max(e) for e in edges) + 1
    imshapes = [None] * n_imgs
    for e, (i, j) in enumerate(edges):
        shape_i = tuple(pred_i[e].shape[0:2])
        shape_j = tuple(pred_j[e].shape[0:2])
        if imshapes[i]:
            assert imshapes[i] == shape_i, f"incorrect shape for image {i}"
        if imshapes[j]:
            assert imshapes[j] == shape_j, f"incorrect shape for image {j}"
        imshapes[i] = shape_i
        imshapes[j] = shape_j
    return imshapes


def get_conf_trf(mode):
    """Get conf trf.

    Args:
        mode: The mode.
    """
    if mode == "log":

        def conf_trf(x):
            """Conf trf.

            Args:
                x: The x.
            """
            return x.log()

    elif mode == "sqrt":

        def conf_trf(x):
            """Conf trf.

            Args:
                x: The x.
            """
            return x.sqrt()

    elif mode == "m1":

        def conf_trf(x):
            """Conf trf.

            Args:
                x: The x.
            """
            return x - 1

    elif mode in ("id", "none"):

        def conf_trf(x):
            """Conf trf.

            Args:
                x: The x.
            """
            return x

    else:
        raise ValueError(f"bad mode for {mode=}")
    return conf_trf


def l2_dist(a, b, weight):
    """L2 dist.

    Args:
        a: The a.
        b: The b.
        weight: The weight.
    """
    return (a - b).square().sum(dim=-1) * weight


def l1_dist(a, b, weight):
    """L1 dist.

    Args:
        a: The a.
        b: The b.
        weight: The weight.
    """
    return (a - b).norm(dim=-1) * weight


ALL_DISTS = dict(l1=l1_dist, l2=l2_dist)


def signed_log1p(x):
    """Signed log1p.

    Args:
        x: The x.
    """
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))


def signed_expm1(x):
    """Signed expm1.

    Args:
        x: The x.
    """
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))


def cosine_schedule(t, lr_start, lr_end):
    """Cosine schedule.

    Args:
        t: The t.
        lr_start: The lr start.
        lr_end: The lr end.
    """
    assert 0 <= t <= 1
    return lr_end + (lr_start - lr_end) * (1 + np.cos(t * np.pi)) / 2


def linear_schedule(t, lr_start, lr_end):
    """Linear schedule.

    Args:
        t: The t.
        lr_start: The lr start.
        lr_end: The lr end.
    """
    assert 0 <= t <= 1
    return lr_start + (lr_end - lr_start) * t
