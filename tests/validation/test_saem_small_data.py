"""SAEM on a small dataset with a weakly determined parameter: with compat="r"
(saemix's ceiling(50/N) chains) no seed may drift off.

The model is a stand-in for a celiac-disease model (21 patients, two
observations each) where one sensitivity saturates: once a subject's sample of
it is near zero, the predictions no longer depend on it. With one chain, the
variance of its random effect could grow during annealing and the fixed effect
drift (about 20-30% of seeds ended with an SD above 5 on the log scale); with
3 chains no seed did (out of 100).
"""

import sys
import warnings

import numpy as np
import pytest

from pynlme import WeakIdentificationWarning, nlmefitsa

pytestmark = pytest.mark.slow


def model(phi, X, V):
    phi = np.atleast_2d(phi)
    s1, s2 = phi[:, 0], phi[:, 1]
    dose, feature = X[:, 0], X[:, 1]
    return np.where(feature == 0, np.log10(1 + s1 * dose), 1.0 / (1.0 + s1 * s2 * dose))


def dataset():
    rng = np.random.default_rng(12345)
    m = 21
    L = np.linalg.cholesky(np.array([[1.0, 1.5], [1.5, 4.0]]))
    eta = rng.standard_normal((m, 2)) @ L.T + np.log([0.4, 1.0])
    dose = rng.uniform(1, 6, m)
    X = np.column_stack([np.repeat(dose, 2), np.tile([0, 1], m)])
    phi = np.repeat(np.exp(eta), 2, axis=0)
    y = model(phi, X, None) + 0.15 * rng.standard_normal(2 * m)
    return X, y, np.repeat(np.arange(m), 2)


def fit(seed, **kwargs):
    X, y, group = dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # overflow in exp
        _, psi, _, _ = nlmefitsa(
            X,
            y,
            group,
            None,
            model,
            np.log([0.4, 1.0]),
            ParamTransform=[1, 1],
            Vectorization="Full",
            CovPattern=np.ones((2, 2)),
            compat="r",
            random_state=seed,
            **kwargs,
        )
    return np.sqrt(np.diag(psi))


def test_no_seed_drifts_with_saemix_chains():
    with warnings.catch_warnings():
        warnings.simplefilter("error", WeakIdentificationWarning)
        sds = [fit(seed) for seed in range(20)]
    assert np.max(sds) < 5, np.round(sds, 2)


@pytest.mark.parametrize("limiter", [False, True])
def test_one_chain_divergence(monkeypatch, capsys, limiter):
    """Reports how often one chain drifts off (SD above 5) in 100 seeds, without
    and with pynlme's variance limiter (run with -s to see the counts). One chain
    is below saemix's minimum, so the chain rule is bypassed here."""
    nlmefit_module = sys.modules["pynlme.nlmefit"]
    monkeypatch.setattr(nlmefit_module, "_saem_chains", lambda n, m, compat: 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WeakIdentificationWarning)
        sds = np.array([fit(seed, VarianceLimiter=limiter) for seed in range(100)])
    drifted = int(np.sum(sds.max(axis=1) > 5))
    with capsys.disabled():
        print(
            f"\ncompat='r', 1 chain, variance_limiter={limiter}: "
            f"{drifted}/100 seeds with an SD above 5"
        )
    assert drifted < 100
