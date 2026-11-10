"""nlmefitsa with compat="r": saemix's chain count, its SAEM details, and the
warning for parameters the data barely determine."""

import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import pynlme
import pynlme.lindstrom_bates as lb
from pynlme import WeakIdentificationWarning, nlmefitsa
from pynlme.r_saemix import _iterations

# pynlme.nlmefit is also the name of a function, so take the module itself.
nlmefit_module = sys.modules["pynlme.nlmefit"]


def small_problem(m=12, seed=0):
    """Exponential decay plus a baseline; 3 log-normal parameters, m subjects."""

    def model(phi, X, V):
        phi = np.atleast_2d(phi)
        return phi[:, 0] * np.exp(-phi[:, 1] * X[:, 0]) + phi[:, 2]

    rng = np.random.default_rng(seed)
    t = np.tile(np.linspace(0, 3, 6), m)
    group = np.repeat(np.arange(m), 6)
    phi = np.exp(np.log([5, 1, 0.5]) + 0.2 * rng.standard_normal((m, 3)))[group]
    y = model(phi, t[:, None], None) + 0.05 * rng.standard_normal(len(t))
    return t[:, None], y, group, model, np.log([5, 1, 0.5])


@pytest.mark.parametrize(
    "n_chains, m, compat, expected",
    [
        (None, 21, "r", 3),  # saemix: ceiling(50/N)
        (None, 12, "r", 5),
        (None, 50, "r", 1),
        (None, 200, "r", 1),
        (4, 21, "r", 4),  # more than the minimum is kept
        (None, 21, "matlab", 5),  # MATLAB: max(1, round(100/m))
        (None, 300, "matlab", 1),
        (1, 21, "matlab", 1),
    ],
)
def test_chain_count(n_chains, m, compat, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert nlmefit_module._saem_chains(n_chains, m, compat) == expected


def test_fewer_chains_than_saemix_are_raised_with_a_warning():
    with pytest.warns(UserWarning, match="increasing the number of chains from 1 to 3"):
        assert nlmefit_module._saem_chains(1, 21, "r") == 3


def test_nlmefitsa_r_raises_one_chain(monkeypatch):
    X, y, group, model, beta0 = small_problem()
    used = []
    run = nlmefit_module._run_saem

    def spy(*args, **kwargs):
        used.append((kwargs["n_chains"], kwargs["saemix"]))
        return run(*args, **kwargs)

    monkeypatch.setattr(nlmefit_module, "_run_saem", spy)
    options = {"ParamTransform": [1, 1, 1], "NIterations": [5, 5, 5], "random_state": 1}
    with pytest.warns(UserWarning, match="from 1 to 5"):
        nlmefitsa(X, y, group, None, model, beta0, compat="r", NChains=1, **options)
    nlmefitsa(X, y, group, None, model, beta0, compat="r", **options)
    nlmefitsa(X, y, group, None, model, beta0, **options)
    assert used == [(5, True), (5, True), (8, False)]


def test_saemix_wrapper_leaves_the_chain_count_to_the_rule():
    assert _iterations({})["n_chains"] is None
    assert _iterations({"nb.chains": 2})["n_chains"] == 2


@pytest.mark.skipif(lb._kernels is None, reason="compiled kernels not built")
@pytest.mark.parametrize("error_model", ["constant", "proportional", "combined"])
def test_saemix_mode_kernel_same_chain(error_model):
    """saemix's E-step (block moves, scales adapted once per iteration) with the
    Rust kernel takes the same chain as the NumPy code. Three random effects, so
    the third kernel's block size cycles between 2 and 3."""
    X, y, group, model, beta0 = small_problem()

    def run():
        return nlmefitsa(
            X,
            y,
            group,
            None,
            model,
            beta0,
            ParamTransform=[1, 1, 1],
            ErrorModel=error_model,
            NIterations=[20, 10, 10],
            compat="r",
            random_state=2,
        )

    kernels = lb._kernels
    try:
        lb._kernels = None
        beta_np, psi_np, stats_np, _ = run()
    finally:
        lb._kernels = kernels
    beta_rs, psi_rs, stats_rs, _ = run()
    np.testing.assert_allclose(beta_rs, beta_np, rtol=1e-9)
    np.testing.assert_allclose(psi_rs, psi_np, rtol=1e-9)
    np.testing.assert_allclose(stats_rs.errorparam, stats_np.errorparam, rtol=1e-9)


def test_saemix_mode_recovers_the_parameters():
    X, y, group, model, beta0 = small_problem(m=20)
    beta, psi, stats, _ = nlmefitsa(
        X,
        y,
        group,
        None,
        model,
        beta0,
        ParamTransform=[1, 1, 1],
        compat="r",
        random_state=3,
    )
    np.testing.assert_allclose(np.exp(beta), [5, 1, 0.5], rtol=0.15)
    assert np.all(np.sqrt(np.diag(psi)) < 0.5)
    assert 0.03 < stats.errorparam[0] < 0.08


def _fake_model(transform, m=4):
    """What _warn_weak_identification reads: an intercept per parameter, all
    with random effects."""
    p = len(transform)
    design = np.tile(np.eye(p), (m, 1, 1))
    return SimpleNamespace(transform=np.asarray(transform), A=design, B=design)


@pytest.mark.parametrize(
    "transform, sd, se, expected",
    [
        (
            [1, 1],
            [1.0, 30.0],
            None,
            ["phi[:, 1]", "SD of its random effect is 30", "orders of magnitude"],
        ),
        ([1, 3], [1.0, 7.0], None, ["phi[:, 1]", "logit scale"]),
        (
            [1, 1],
            [1.0, 2.0],
            [0.1, 9.0],
            ["phi[:, 1]", "standard error of its fixed effect is 9"],
        ),
        (
            [1, 1],
            [1.0, 2.0],
            [0.1, np.inf],
            ["standard error of its fixed effect is inf"],
        ),
    ],
)
def test_weak_identification_warns(transform, sd, se, expected):
    psi = np.diag(np.square(sd))
    with pytest.warns(WeakIdentificationWarning) as record:
        nlmefit_module._warn_weak_identification(_fake_model(transform), psi, se)
    assert len(record) == 1
    for text in expected:
        assert text in str(record[0].message)


@pytest.mark.parametrize(
    "transform, sd, se",
    [
        ([1, 1], [1.0, 5.9], [0.5, 5.9]),  # below the limits
        ([0, 1], [100.0, 1.0], [50.0, 0.5]),  # identity scale: not judged
    ],
)
def test_weak_identification_quiet(transform, sd, se):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nlmefit_module._warn_weak_identification(
            _fake_model(transform), np.diag(np.square(sd)), se
        )


def test_fit_names_the_parameter(monkeypatch):
    """fit() re-issues the warning with the parameter's name."""
    X, y, group, model, beta0 = small_problem()
    real = pynlme.api.nlmefitsa

    def warn_then_fit(*args, **kwargs):
        warnings.warn(
            WeakIdentificationWarning("parameter {name} is weak", 1), stacklevel=2
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(pynlme.api, "nlmefitsa", warn_then_fit)
    data = {"t": X[:, 0], "y": y, "id": group}
    with pytest.warns(WeakIdentificationWarning, match="parameter 'k' is weak"):
        pynlme.fit(
            lambda p, x: p["a"] * np.exp(-p["k"] * x["t"]) + p["c"],
            data,
            response="y",
            group="id",
            predictors=["t"],
            start={"a": 5.0, "k": 1.0, "c": 0.5},
            transform={"a": "log", "k": "log", "c": "log"},
            method="SAEM",
            n_iterations=[5, 5, 5],
            random_state=1,
        )


@pytest.mark.parametrize(
    "compat, option, limiter, extensions",
    [
        ("r", None, False, []),  # plain compat="r": saemix's algorithm
        ("r", True, True, ["variance_limiter"]),  # saemix plus pynlme's limiter
        ("matlab", None, True, None),  # the default mode keeps it
        ("matlab", False, False, None),
    ],
)
def test_variance_limiter(monkeypatch, compat, option, limiter, extensions):
    from pynlme import saem

    X, y, group, model, beta0 = small_problem()
    calls = []
    limit = saem._limit_eigenvalue_decrease

    def counted(new, old):
        calls.append(1)
        return limit(new, old)

    monkeypatch.setattr(saem, "_limit_eigenvalue_decrease", counted)
    kwargs = {} if option is None else {"VarianceLimiter": option}
    _, _, stats, _ = nlmefitsa(
        X,
        y,
        group,
        None,
        model,
        beta0,
        ParamTransform=[1, 1, 1],
        NIterations=[5, 5, 5],
        compat=compat,
        random_state=1,
        **kwargs,
    )
    assert (len(calls) == 5) == limiter and (len(calls) == 0) == (not limiter)
    assert stats.pynlme_extensions == extensions


def test_default_mode_unchanged():
    """The default mode gives what it gave before compat="r" followed saemix
    (values from pynlme 0.5.1)."""
    X, y, group, model, beta0 = small_problem()
    expected = {
        "diagonal": (
            [1.6004390976354712, -0.06895641173346127, -0.9790610395619285],
            [0.04781056475517994, 0.019642478675218327, 0.14766053706370416],
            0.08094976454787821,
        ),
        "full": (
            [1.5960873737640153, -0.05173293168538753, -0.8977899740992695],
            [0.04779707462868376, 0.022472043050100098, 0.14053072550164009],
            0.07645104680939445,
        ),
    }
    for pattern, (beta_ref, psi_ref, error_ref) in expected.items():
        kwargs = {"CovPattern": np.ones((3, 3))} if pattern == "full" else {}
        beta, psi, stats, _ = nlmefitsa(
            X,
            y,
            group,
            None,
            model,
            beta0,
            ParamTransform=[1, 1, 1],
            NIterations=[20, 10, 10],
            random_state=2,
            **kwargs,
        )
        np.testing.assert_allclose(beta, beta_ref, rtol=1e-8)
        np.testing.assert_allclose(np.diag(psi), psi_ref, rtol=1e-8)
        np.testing.assert_allclose(stats.errorparam, [error_ref], rtol=1e-8)
        assert stats.pynlme_extensions is None
