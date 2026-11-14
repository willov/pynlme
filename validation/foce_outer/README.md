# FOCE outer optimiser

FOCE's outer problem (fixed effects and random-effects covariance) is optimised by
L-BFGS-B with a gradient by central differences of the objective (relative step
1e-5); every objective evaluation re-solves all subjects' conditional modes. With
SUND ODE models this needs `rel_tol=1e-12` (see the
[scaling benchmark](../scaling/README.md)): SUND's solution is not smooth in the
parameters at the tolerance level, and the differences amplify that. `run.py`
compares, on models with more parameters than the scaling benchmark's:

- L-BFGS-B, step 1e-5 (the default), at `rel_tol` 1e-12 and 1e-10;
- L-BFGS-B, step 5e-4 (about the cube root of the solution's noise at 1e-10, the
  step that balances noise and truncation error), at 1e-10;
- SciPy's derivative-free COBYQA (quadratic models from objective values, trust
  region; the BOBYQA family, as nlmixr2's default `bobyqa` outer optimiser), at
  1e-10 and 1e-12. In `fit_first_order` as `outer="cobyqa"`.

Cases (simulated, 30 subjects, the same data for every method; SUND, all cores):

- **pkpd**: oral dose, Michaelis-Menten elimination, Emax effect
  (`validation/models/mm_pkpd_sund.txt`); log concentration and log effect observed
  (20 observations per subject); 6 fixed effects, random effects on ka, V, vmax and
  emax: 10 outer parameters with a diagonal covariance, **pkpd_full** 16 with a
  full one. Adams.
- **tmdd4**: the stiff TMDD model (`validation/models/tmdd_sund.txt`) with kel,
  kint, R0 and kdeg estimated, random effects on kel, kint and R0 (7 outer
  parameters); log free drug observed. BDF.

## Results

Seconds per fit, objective evaluations, and log-likelihood difference from the
best fit of the case (2026-11-14, AMD Ryzen 7 7800X3D, idle, pynlme 0.5.2 with the
warm-start fix below, SUND 2.14.1, SciPy 1.18.1):

| Case (outer parameters) | L-BFGS-B, 1e-5, at 1e-12 | L-BFGS-B, 1e-5, at 1e-10 | L-BFGS-B, 5e-4, at 1e-10 | COBYQA at 1e-10 | COBYQA at 1e-12 |
|---|---|---|---|---|---|
| pkpd (10) | 180 s, 1618, -0.000 | 137 s, 1975, **-0.23** | 112 s, 1534, -0.000 | **23 s**, 288, -0.000 | 39 s, 272, 0 |
| pkpd_full (16) | 463 s, 4291, -0.000 | 172 s, 2410, **-0.98** | 104 s, 1453, -0.010 | **62 s**, 799, -0.007 | 136 s, 1014, 0 |
| tmdd4 (7) | 195 s, 1201, -0.000 | 131 s, 1306, **-4.82** | 57 s, 541, -0.000 | **22 s**, 209, -0.001 | 43 s, 238, 0 |

- The default (L-BFGS-B, step 1e-5) needs 1e-12, as on the scaling benchmark's
  models; at 1e-10 it stops early (fixed effects up to 0.06 off on the log scale,
  variances up to 84 % off on tmdd4).
- COBYQA at 1e-10 reaches the optimum (within 0.01 in log-likelihood; fixed
  effects within 0.003, variances within 3 %) with about 5.5 times fewer objective
  evaluations than the default at 1e-12, and is 7.5 to 9 times faster. At 1e-12 it
  reaches the best fit of every case. Its evaluations grow with the number of
  parameters (288 for 10, 799 for 16), as do L-BFGS-B's.
- The larger difference step lets L-BFGS-B work at 1e-10 too (1.6 to 4.5 times
  faster than the default at 1e-12), but it needs 1.8 to 5.3 times COBYQA's
  evaluations.

## Warm start of the inner problem

Before this branch each objective evaluation warm-started the subjects' conditional
modes from the last evaluation's. After a far trial point (L-BFGS-B's first line
search step) those modes were a poor start, the inner iteration ended elsewhere, and
the objective at the same parameters changed: on pkpd -187.4 at the start, -115.0
after one iteration at the same point, so L-BFGS-B stopped there ("converged",
log-likelihood 115 at 1e-10 and 134 at 1e-12 against the optimum's 251; pkpd_full
141 against 255). COBYQA, whose trust region keeps trial points close, was not
affected. The fix warm-starts from the modes of the best point so far.

## Running

```bash
uv run python validation/foce_outer/run.py [pkpd pkpd_full tmdd4] [--subjects 30]
```

writes `results.json` (about 30 minutes for all).
