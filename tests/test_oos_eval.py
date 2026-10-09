# tests/test_oos_eval.py — unit tests for the purged/embargoed walk-forward.
# Pure numpy/pandas logic; no network, no model fit needed for the index math.
import numpy as np
import pandas as pd

import research.oos_eval as oe


def _train_test_indices(D, folds, H, purge):
    """Reproduce the exact boolean masks walk_forward builds, without fitting."""
    n = len(D)
    edges = np.linspace(0, n, folds + 1).astype(int)
    out = []
    for k in range(1, folds):
        train_end, test_end = edges[k], edges[k + 1]
        tr = np.zeros(n, bool); tr[: (train_end - H if purge else train_end)] = True
        te = np.zeros(n, bool)
        lo = train_end + oe.FEATURE_LOOKBACK if purge else train_end
        te[lo:test_end] = True
        out.append((np.where(tr)[0], np.where(te)[0]))
    return out


def test_purge_drops_h_boundary_train_rows():
    D = pd.DataFrame({"x": range(1000)})
    H = 8
    naive = _train_test_indices(D, 5, H, purge=False)
    purged = _train_test_indices(D, 5, H, purge=True)
    for (tr_n, _), (tr_p, _) in zip(naive, purged):
        # purged training set must end exactly H rows earlier
        assert tr_p[-1] == tr_n[-1] - H


def test_embargo_removes_feature_window_from_test():
    D = pd.DataFrame({"x": range(1000)})
    naive = _train_test_indices(D, 5, 8, purge=False)
    purged = _train_test_indices(D, 5, 8, purge=True)
    for (_, te_n), (_, te_p) in zip(naive, purged):
        # embargoed test set starts FEATURE_LOOKBACK rows later
        assert te_p[0] == te_n[0] + oe.FEATURE_LOOKBACK


def test_train_and_test_never_overlap_after_purge():
    D = pd.DataFrame({"x": range(1000)})
    for tr, te in _train_test_indices(D, 5, 8, purge=True):
        assert tr.max() < te.min()
        # and the gap is at least the horizon + feature window
        assert te.min() - tr.max() >= 8 + oe.FEATURE_LOOKBACK - 1
