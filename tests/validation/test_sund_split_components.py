"""
sund_model with separate integrators per subject (split_components=True, in parallel)
and with coupled batches (split_components=False).
"""

import os
from pathlib import Path

import numpy as np
import pytest

sund = pytest.importorskip("sund")
from pynlme import sund_model  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "validation" / "models"
PARAMS = ["ka", "V", "vmax", "km"]
TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
N = 12


@pytest.fixture(scope="module", autouse=True)
def installed():
    sund.install_model(os.path.relpath(MODELS / "mm_pk_sund.txt"))


def subjects(n=N, times=None):
    """(phi, x, v) with one row per observation, each subject on its own parameters."""
    rng = np.random.default_rng(0)
    phi = np.column_stack(
        [
            np.full(n, 1.0),
            10 * np.exp(0.2 * rng.standard_normal(n)),
            2 * np.exp(0.25 * rng.standard_normal(n)),
            np.full(n, 1.5),
        ]
    )
    times = [TIMES] * n if times is None else times
    rows = np.repeat(np.arange(n), [len(t) for t in times])
    return phi[rows], np.concatenate(times)[:, None], np.full((len(rows), 1), 100.0)


def model(**options):
    return sund_model("mm_pk", "conc", PARAMS, v_states=["gut"], **options)


def test_split_solution_does_not_depend_on_batch_or_threads():
    phi, x, v = subjects()
    reference = model(batch_size=1, threads=1)(phi, x, v)
    for options in (
        {"batch_size": 5, "threads": 1},
        {"batch_size": N, "threads": 1},
        {"batch_size": N, "threads": 4},
        {},
    ):
        np.testing.assert_array_equal(model(**options)(phi, x, v), reference)


def test_coupled_batches_agree_within_the_tolerance():
    phi, x, v = subjects()
    split = model()(phi, x, v)
    coupled = model(split_components=False)(phi, x, v)
    np.testing.assert_allclose(coupled, split, rtol=1e-7)
    assert not np.array_equal(coupled, split)  # they are different integrations


def test_different_sampling_times_per_subject():
    rng = np.random.default_rng(1)
    times = [np.sort(TIMES * np.exp(0.1 * rng.standard_normal(6))) for _ in range(N)]
    phi, x, v = subjects(times=times)
    one_by_one = model(batch_size=1)(phi, x, v)
    np.testing.assert_allclose(model()(phi, x, v), one_by_one, rtol=1e-7)


def test_split_solution_does_not_depend_on_other_subjects_parameters():
    rng = np.random.default_rng(1)
    mixed = [np.sort(TIMES * np.exp(0.1 * rng.standard_normal(6))) for _ in range(N)]
    for times in (None, mixed):
        phi, x, v = subjects(times=times)
        fn = model(batch_size=N)
        reference = fn(phi, x, v)
        changed = phi.copy()
        changed[6:, 2] *= 2  # the other subjects' vmax; each subject has 6 rows
        out = fn(changed, x, v)
        np.testing.assert_array_equal(out[:6], reference[:6])
        assert not np.array_equal(out[6:], reference[6:])


def failing_row_values(**options):
    """The predictions per subject with subject 3's integration failing (an infinite
    Vmax: CVODE reports a convergence failure)."""
    phi, x, v = subjects()
    phi = phi.copy()
    phi[x[:, 0].size // N * 3 : x[:, 0].size // N * 4, 2] = np.inf
    return model(**options)(phi, x, v).reshape(N, -1)


def test_failure_only_affects_that_subject_when_split():
    out = failing_row_values(batch_size=N)
    assert np.all(~np.isfinite(out[3]))
    assert np.all(np.isfinite(np.delete(out, 3, axis=0)))


def test_failure_affects_the_whole_coupled_batch():
    out = failing_row_values(batch_size=N, split_components=False)
    assert np.all(~np.isfinite(out))


def test_simulation_is_usable_after_a_failure():
    phi, x, v = subjects()
    fn = model(batch_size=N)
    expected = fn(phi, x, v)
    failing = phi.copy()
    failing[:, 2] = np.inf
    assert np.all(~np.isfinite(fn(failing, x, v)))
    np.testing.assert_array_equal(fn(phi, x, v), expected)


def test_threads_are_only_set_during_the_simulation():
    phi, x, v = subjects()
    sund.set_number_of_threads(1)
    fn = model(threads=4)
    assert sund.get_number_of_threads() == 1
    fn(phi, x, v)
    assert sund.get_number_of_threads() == 1


def test_invalid_threads_raise_when_building_the_model():
    with pytest.raises((TypeError, ValueError)):
        model(threads="many")


def test_default_threads_follow_the_environment(monkeypatch):
    # SUND reads the variable when asked for its default, on each call with None.
    monkeypatch.setenv("SUND_NUM_THREADS", "2")
    seen = []
    original = sund.set_number_of_threads

    def record(value):
        seen.append(value)
        original(value)

    monkeypatch.setattr(sund, "set_number_of_threads", record)
    phi, x, v = subjects()
    model()(phi, x, v)
    assert None in seen
    assert "all" not in seen
