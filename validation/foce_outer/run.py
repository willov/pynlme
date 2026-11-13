"""FOCE outer optimiser experiment: L-BFGS-B with a central-difference gradient
(the default, step 1e-5 or larger) against the derivative-free COBYQA, with SUND
ODE models at rel_tol 1e-10 and 1e-12, on models with more parameters than the
scaling benchmark's:

- pkpd: oral dose, Michaelis-Menten elimination, Emax effect
  (validation/models/mm_pkpd_sund.txt); log concentration and log effect observed;
  6 fixed effects, random effects on ka, V, vmax and emax (diagonal: 10 outer
  parameters; full: 16).
- tmdd4: the stiff TMDD model (validation/models/tmdd_sund.txt) with kel, kint, R0
  and kdeg estimated, random effects on kel, kint and R0 (7 outer parameters).

Run from the repository root (writes validation/foce_outer/results.json):

    uv run python validation/foce_outer/run.py [case ...] [--subjects N]
"""

import argparse
import functools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pynlme
import pynlme.first_order as first_order
from pynlme import nlmefit

# The module, not the function the package exports under the same name.
nlmefit_module = sys.modules["pynlme.nlmefit"]

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
RESULTS = HERE / "results.json"

PKPD = {
    "parameters": ["ka", "V", "vmax", "km", "emax", "ec50"],
    "truth": np.array([1.0, 10.0, 1.0, 2.0, 100.0, 2.0]),
    "sd": np.array([0.3, 0.2, 0.25, 0.0, 0.2, 0.0]),
    "random": [1, 2, 3, 5],  # 1-based, as REParamsSelect
    "start": np.array([1.3, 0.8, 1.2, 0.8, 1.2, 0.8]),
    "times": [0.5, 1, 2, 3, 4, 6, 8, 10, 12],
    "doses": [20.0, 40.0, 80.0],
    "sigma": 0.1,
}
TMDD4 = {
    "parameters": ["kel", "kint", "R0", "kdeg"],
    "truth": np.array([0.1, 0.05, 10.0, 0.1]),
    "sd": np.array([0.3, 0.3, 0.2, 0.0]),
    "random": [1, 2, 3],
    "start": np.array([1.5, 1.4, 0.8, 0.7]),
    "times": [0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96],
    "doses": [1.0, 3.0, 10.0, 100.0],
    "sigma": 0.1,
}


def pkpd_model(rel_tol, threads=None):
    import sund

    from pynlme import sund_model

    sund.install_model(os.path.relpath(MODELS / "mm_pkpd_sund.txt"))
    inner = sund_model(
        "mm_pkpd",
        ["conc", "effect"],
        PKPD["parameters"],
        v_states=["gut"],
        rel_tol=rel_tol,
        abs_tol=rel_tol * 1e-2,
        method="adams",
        threads=threads,
    )
    return lambda phi, x, v: np.log(inner(phi, x, v))


def tmdd4_model(rel_tol, threads=None):
    import sund

    from pynlme import sund_model

    sund.install_model(os.path.relpath(MODELS / "tmdd_sund.txt"))
    return sund_model(
        "tmdd",
        "logL",
        TMDD4["parameters"],
        v_states=["L"],
        rel_tol=rel_tol,
        abs_tol=rel_tol * 1e-2,
        threads=threads,
    )


def simulate(case, n, threads=None):
    """Data from the case's own model at rel_tol 1e-12 (fixed seed)."""
    spec = PKPD if case.startswith("pkpd") else TMDD4
    model = (pkpd_model if case.startswith("pkpd") else tmdd4_model)(1e-12, threads)
    rng = np.random.default_rng(2026 + n)
    t = np.array(spec["times"], dtype=float)
    X, y, group, dose = [], [], [], []
    for i in range(n):
        phi = spec["truth"] * np.exp(rng.normal(size=spec["truth"].size) * spec["sd"])
        d = rng.choice(spec["doses"])
        if case.startswith("pkpd"):
            x = np.column_stack([np.tile(t, 2), np.repeat([0, 1], t.size)])
        else:
            x = t[:, None]
        f = model(np.tile(phi, (len(x), 1)), x, np.array([[d]]).repeat(len(x), 0))
        X.append(x)
        y.append(f + spec["sigma"] * rng.normal(size=f.size))
        group.append(np.full(len(f), i))
        dose.append(d)
    return (
        np.vstack(X),
        np.concatenate(y),
        np.concatenate(group),
        np.array(dose)[:, None],
    )


class Counter:
    """Counts objective evaluations (one linearized_marginal call each)."""

    def __init__(self):
        self.n = 0
        self.original = first_order.linearized_marginal

    def __enter__(self):
        def counted(*args, **kwargs):
            self.n += 1
            return self.original(*args, **kwargs)

        first_order.linearized_marginal = counted
        return self

    def __exit__(self, *exc):
        first_order.linearized_marginal = self.original


def fit(case, data, rel_tol, outer, fd_step, threads=None):
    spec = PKPD if case.startswith("pkpd") else TMDD4
    model = (pkpd_model if case.startswith("pkpd") else tmdd4_model)(rel_tol, threads)
    X, y, group, dose = data
    full = case.endswith("full")
    cov = {"CovPattern": np.ones((len(spec["random"]),) * 2)} if full else {}
    nlmefit_module.fit_first_order = functools.partial(
        first_order.fit_first_order, outer=outer, fd_step=fd_step
    )
    try:
        with Counter() as counter:
            start = time.perf_counter()
            beta, psi, stats, _ = nlmefit(
                X,
                y,
                group,
                dose,
                model,
                np.log(spec["truth"] * spec["start"]),
                ParamTransform=[1] * spec["truth"].size,
                REParamsSelect=spec["random"],
                ApproximationType="FOCE",
                Vectorization="Full",
                **cov,
            )
            seconds = time.perf_counter() - start
    finally:
        nlmefit_module.fit_first_order = first_order.fit_first_order
    return {
        "time_s": seconds,
        "logl": float(stats.logl),
        "evaluations": counter.n,
        "beta": np.asarray(beta).tolist(),
        "psi_diag": np.diag(psi).tolist(),
    }


# (label, rel_tol, outer, fd_step)
METHODS = [
    ("L-BFGS-B, step 1e-5, 1e-12", 1e-12, "lbfgsb", 1e-5),
    ("L-BFGS-B, step 1e-5, 1e-10", 1e-10, "lbfgsb", 1e-5),
    ("L-BFGS-B, step 5e-4, 1e-10", 1e-10, "lbfgsb", 5e-4),
    ("COBYQA, 1e-10", 1e-10, "cobyqa", None),
    ("COBYQA, 1e-12", 1e-12, "cobyqa", None),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=["pkpd", "pkpd_full", "tmdd4"])
    parser.add_argument("--subjects", type=int, default=30)
    parser.add_argument("--threads", default=None)
    parser.add_argument("--methods", default=None, help="comma-separated indices")
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()
    threads = int(args.threads) if args.threads else None
    methods = (
        [METHODS[int(k)] for k in args.methods.split(",")] if args.methods else METHODS
    )
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    for case in args.cases:
        data = simulate(case, args.subjects, threads)
        key = f"{case}_{args.subjects}"
        entry = results.setdefault(key, {})
        for label, rel_tol, outer, fd_step in methods:
            load = os.getloadavg()
            try:
                r = fit(case, data, rel_tol, outer, fd_step or 1e-5, threads)
            except Exception as error:  # noqa: BLE001 - recorded
                r = {"error": repr(error)}
            r["load_before"] = load
            entry[label] = r
            print(
                f"{key:14} {label:28} "
                + (
                    f"{r['time_s']:8.1f} s  logl {r['logl']:.4f}  evals {r['evaluations']}"
                    if "logl" in r
                    else r["error"]
                ),
                flush=True,
            )
            out.write_text(json.dumps(results, indent=1))
    import scipy
    import sund

    results["_environment"] = {
        "python": platform.python_version(),
        "pynlme": pynlme.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sund": sund.__version__,
        "cpus": os.cpu_count(),
        "threads": args.threads or "all cores",
    }
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
