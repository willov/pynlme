# Scaling benchmark

How the fitting time grows with the number of subjects (10, 100, 1000), for a
closed-form model and three ODE models (non-stiff, multiple doses, stiff), in pynlme,
MATLAB R2023b, R (`nlme`, `saemix`) and nlmixr2.

- **Data**: simulated by `make_data.py` (fixed seeds), the same files for every
  program. *oral*: one-compartment model with first-order absorption (as
  Theophylline), random effects on log ka and log CL. *mm*: oral dose with
  Michaelis-Menten elimination, an ODE without a closed form, random effects on log V
  and log Vmax. *mmd*: the mm model with doses at 0, 12 and 24 h. *tmdd*: target-mediated
  drug disposition (`validation/models/tmdd_sund.txt`), stiff (binding about 1000 times
  faster than elimination), log free drug observed, random effects on log kel and log
  kint. Additive error, the same start values everywhere.
- **Options**: every program with its defaults (nlmefit/nlmefitsa's, `nlmeControl()`,
  saemix's, nlmixr2's; pynlme's MATLAB mode follows MATLAB's defaults, its R mode
  nlme's).
- **ODE solvers**: every program at the same tolerance, relative 1e-8 and absolute
  1e-10, with its usual method for the model: for mm and mmd the non-stiff one (SUND
  Adams and SciPy DOP853 in pynlme, `ode45` in MATLAB, deSolve's `lsoda` in `nlme`,
  rxode2 in nlmixr2), for tmdd the stiff one (SUND BDF, SciPy LSODA, `ode15s`,
  `lsoda`, rxode2). At this setting the solvers are about equally accurate (table
  below). One rule for all programs: a fit that fails or ends at a wrong optimum at
  the common tolerance is run at a tighter one and marked † with its relative
  tolerance. This applies to pynlme's MATLAB mode with SUND, run at 1e-10, FOCE with
  SUND, run at 1e-12 (on mmd and tmdd the tables still show the run at 1e-10, which
  stops slightly early; re-timing at 1e-12 is pending), and R `nlme` on tmdd, whose PNLS step fails near the optimum
  at 1e-6 to 1e-10, run at 1e-12. At 1e-10 for all, `nlme` instead took a wrong path
  on mm, so no single tolerance suits every program. SUND needs the tighter tolerance
  in the fits that take finite differences of the solution because its solution is
  not smooth in the parameters: changing a parameter by up to 1e-4 moves the mm
  solution off a smooth curve by about the tolerance (8e-10 of the largest
  concentration at 1e-10, 2e-12 at 1e-12), against 8e-13 for SciPy's DOP853 at 1e-8,
  and the finite differences amplify that. At 1e-8 the mm fit in the MATLAB mode ends
  at a wrong optimum. FOCE, whose outer gradient takes central differences of the
  objective, stops early at 1e-10 on every ODE model: on mm at 10 subjects
  log-likelihood -0.09 (1.15 at 1e-12 and with SciPy), with each subject's own
  integrator and with SUND's older coupled batches (0.71) alike; on tmdd at 100
  subjects 481.47 (482.68 at 1e-12); on mmd at 100 subjects 271.63 (271.68).
- **Model calls**: the closed-form model is called for all subjects at once in pynlme
  and MATLAB (`Vectorization="Full"`); pynlme passes all subjects to the ODE model at
  once: `sund_model` with its defaults (each subject its own integrator, batches of
  64 subjects run in parallel on all cores), or `ode_model` (all subjects as one
  SciPy system, one thread).
- **Timing**: one machine (AMD Ryzen 7 7800X3D, 8 cores, 16 threads, 32 GB RAM,
  Linux), idle during the runs (load logged; `_environment` in `results/pynlme.json`).
  pynlme 0.5.2 with its Rust kernels, Python 3.14, SciPy 1.18, SUND 2.14.1 (measured
  2026-11-12); MATLAB R2023b Update 1; R 4.6.1, nlme 3.1-169, saemix 3.5, nlmixr2
  7.0.1, deSolve 1.42 (measured 2026-10). SUND runs subjects in parallel on all 16
  threads, SciPy's `ode_model` on one. Runs under 5 s
  are the best of 3; in R and MATLAB the first run of each method is an untimed
  warm-up (pynlme compiles its SUND models before timing). Once a run
  takes over 60 s, the larger sizes are skipped ("skipped": they would take ten times
  longer); for SAEM only from 100 subjects on, as on 10 subjects nlmefitsa runs
  several chains by default. MATLAB's FOCE on tmdd was stopped after 3.3 hours at 10
  subjects, and on mmd not run.

## Solver accuracy

`solver_accuracy.py` (with `.R` and `.m`) solves 20 parameter sets of mm and tmdd
with each program's solver at the common tolerance and compares with a reference at
1e-13 (mm: largest error relative to the largest concentration; tmdd: largest error
in log free drug). SUND solves each parameter set with its own integrator, as
`sund_model` now does by default; its values are the same with SUND 2.5.3 and 2.14.1:

| Solver | mm (non-stiff) | tmdd (stiff) |
|---|---|---|
| SciPy (DOP853 / LSODA), pynlme | 1.9e-9 | 6.0e-7 |
| SUND (Adams / BDF), pynlme | 2.6e-8 | 1.2e-6 |
| deSolve `lsoda`, R `nlme` | 1.7e-8 | 7.2e-7 |
| rxode2, nlmixr2 | 1.7e-8 | 6.0e-7 |
| MATLAB (`ode45` / `ode15s`) | 2.1e-9 | 1.8e-6 |

## Results

Closed-form oral PK model (3 parameters, 2 random effects, 10 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode | 0.006 | 0.028 | 0.265 |
| LME | MATLAB `nlmefit` | 0.012 | 0.081 | 5.58 |
| LME | pynlme, R mode | 0.004 | 0.018 | 0.140 |
| LME | R `nlme` | 0.009 | 0.026 | 0.200 |
| FOCE | pynlme | 0.378 | 1.11 | 6.59 |
| FOCE | MATLAB `nlmefit` | 1.48 | 10 | 599 |
| FOCE | nlmixr2 (FOCEi) | 0.388 | 0.415 | 1.41 |
| SAEM | pynlme `nlmefitsa` | 0.265 | 0.272 | 1.22 |
| SAEM | MATLAB `nlmefitsa` | 0.600 | 0.948 | 3.02 |
| SAEM | pynlme `pynlme.r.saemix` | 0.246 | 0.594 | 3.86 |
| SAEM | R `saemix` | 1.26 | 2.81 | 24 |
| SAEM | nlmixr2 | 1.16 | 8.19 | 104 |

Log-likelihood each program reports, as a check that the fits are equally good (LME, FOCE and saemix's SAEM; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode | -88.67 | -1040.13 | -10337.75 |
| LME | MATLAB `nlmefit` | -88.67 | -1040.13 | -10337.75 |
| LME | pynlme, R mode | -88.67 | -1040.13 | -10337.73 |
| LME | R `nlme` | -88.67 | -1040.13 | -10337.73 |
| FOCE | pynlme | -88.66 | -1040.04 | -10336.82 |
| FOCE | MATLAB `nlmefit` | -89.10 | -1047.07 | -10411.14 |
| FOCE | nlmixr2 (FOCEi) | -88.66 | -1040.04 | -10336.83 |
| SAEM | pynlme `pynlme.r.saemix` | -88.63 | -1040.49 | -10336.23 |
| SAEM | R `saemix` | -88.59 | -1040.19 | -10336.48 |

Fixed effects: largest difference (log scale) from pynlme's R mode LME fit, as a check that the fits agree (also for SAEM, which reports no log-likelihood by default):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode | 0.000 | 0.000 | 0.000 |
| FOCE | pynlme | 0.004 | 0.006 | 0.006 |
| SAEM | pynlme `nlmefitsa` | 0.006 | 0.009 | 0.008 |
| SAEM | pynlme `pynlme.r.saemix` | 0.011 | 0.009 | 0.008 |

Michaelis-Menten ODE model (4 parameters, 2 random effects, 8 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 0.514 †1e-10 | 0.566 †1e-10 | 13 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 0.419 | 0.680 | 4.41 |
| LME | MATLAB `nlmefit` (ode45) | 0.390 | 3.49 | 47 |
| LME | pynlme, R mode (SUND) | 0.057 | 0.201 | 1.60 |
| LME | pynlme, R mode (SciPy) | 0.185 | 0.509 | 4.37 |
| LME | R `nlme` (deSolve) | 1.35 | 13 | 210 |
| FOCE | pynlme (SUND) | 78 †1e-12 | skipped | skipped |
| FOCE | pynlme (SciPy) | 108 | skipped | skipped |
| FOCE | MATLAB `nlmefit` (ode45) | 426 | skipped | skipped |
| FOCE | nlmixr2 (FOCEi, rxode2) | 0.881 | 1.86 | 14 |
| SAEM | pynlme `nlmefitsa` (SUND) | 5.83 | 6.44 | 56 |
| SAEM | pynlme `nlmefitsa` (SciPy) | 23 | 24 | 70 |
| SAEM | MATLAB `nlmefitsa` (ode45) | 135 | 140 | skipped |
| SAEM | nlmixr2 (rxode2) | 5.17 | 37 | 383 |

ODE tolerance relative 1e-8, absolute 1e-10 in every program; † marks a fit run at a tighter relative tolerance (shown), because it fails or misses the optimum at the common one.

Log-likelihood each program reports, as a check that the fits are equally good (LME, FOCE and saemix's SAEM; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 1.10 †1e-10 | -33.91 †1e-10 | -360.26 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 1.10 | -33.91 | -360.28 |
| LME | MATLAB `nlmefit` (ode45) | 1.10 | -33.91 | -360.28 |
| LME | pynlme, R mode (SUND) | 1.10 | -33.90 | -360.22 |
| LME | pynlme, R mode (SciPy) | 1.10 | -33.91 | -360.27 |
| LME | R `nlme` (deSolve) | 1.11 | -33.91 | -360.23 |
| FOCE | pynlme (SUND) | 1.15 †1e-12 | – | – |
| FOCE | pynlme (SciPy) | 1.15 | – | – |
| FOCE | MATLAB `nlmefit` (ode45) | 1.03 | – | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 1.14 | -33.32 | -353.27 |

Fixed effects: largest difference (log scale) from pynlme's R mode (SciPy) LME fit, as a check that the fits agree (also for SAEM, which reports no log-likelihood by default):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 0.001 | 0.000 | 0.000 |
| LME | pynlme, MATLAB mode (SciPy) | 0.001 | 0.000 | 0.000 |
| LME | pynlme, R mode (SUND) | 0.000 | 0.000 | 0.000 |
| FOCE | pynlme (SUND) | 0.068 | – | – |
| FOCE | pynlme (SciPy) | 0.068 | – | – |
| SAEM | pynlme `nlmefitsa` (SUND) | 0.559 | 0.323 | 0.328 |
| SAEM | pynlme `nlmefitsa` (SciPy) | 0.526 | 0.304 | 0.342 |

Michaelis-Menten ODE model with three doses (0, 12 and 24 h; SUND activities, nlmixr2 event table, ode45/deSolve from dose to dose; 15 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | failed | 25 †1e-10 | 361 †1e-10 |
| LME | MATLAB `nlmefit` (ode45) | failed | 31 | 217 |
| LME | pynlme, R mode (SUND) | 0.080 | 0.441 | 4.74 |
| LME | R `nlme` (deSolve) | 5.52 | 55 | 697 |
| FOCE | pynlme (SUND) | failed | 211 †1e-10 | skipped |
| FOCE | MATLAB `nlmefit` (ode45) | not run | not run | not run |
| FOCE | nlmixr2 (FOCEi, rxode2) | 1.40 | 5.09 | 36 |
| SAEM | pynlme `nlmefitsa` (SUND) | 17 | 17 | 155 |
| SAEM | MATLAB `nlmefitsa` (ode45) | 700 | 731 | skipped |
| SAEM | nlmixr2 (rxode2) | 13 | 111 | skipped |

ODE tolerance relative 1e-8, absolute 1e-10 in every program; † marks a fit run at a tighter relative tolerance (shown), because it fails or misses the optimum at the common one.

Log-likelihood each program reports, as a check that the fits are equally good (LME, FOCE and saemix's SAEM; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | – | 271.09 †1e-10 | 2571.64 †1e-10 |
| LME | MATLAB `nlmefit` (ode45) | – | 271.09 | 2571.61 |
| LME | pynlme, R mode (SUND) | 29.87 | 271.08 | 2571.69 |
| LME | R `nlme` (deSolve) | 29.87 | 271.08 | 2571.60 |
| FOCE | pynlme (SUND) | – | 271.63 †1e-10 | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 29.92 | 271.67 | 2576.12 |

Fixed effects: largest difference (log scale) from pynlme's R mode (SUND) LME fit, as a check that the fits agree (also for SAEM, which reports no log-likelihood by default):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | – | 0.000 | 0.000 |
| FOCE | pynlme (SUND) | – | 0.067 | – |
| SAEM | pynlme `nlmefitsa` (SUND) | 0.047 | 0.067 | 0.124 |

Stiff target-mediated drug disposition model (3 parameters, 2 random effects, 3 states, log free drug observed; each program's stiff solver: SUND BDF, SciPy LSODA, ode15s, lsoda; 10 observations per subject); seconds per fit:

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 0.076 †1e-10 | 0.979 †1e-10 | 23 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 3.11 | 28 | 446 |
| LME | MATLAB `nlmefit` (ode15s) | 8.27 | 426 | skipped |
| LME | pynlme, R mode (SUND) | 0.050 | 1.13 | 11 |
| LME | pynlme, R mode (SciPy) | 1.02 | 6.48 | 58 |
| LME | R `nlme` (deSolve) | 7.28 †1e-12 | 84 †1e-12 | skipped |
| FOCE | pynlme (SUND) | 18 †1e-10 | 311 †1e-10 | skipped |
| FOCE | pynlme (SciPy) | 1053 | skipped | skipped |
| FOCE | MATLAB `nlmefit` (ode15s) | > 3 h, stopped | not run | not run |
| FOCE | nlmixr2 (FOCEi, rxode2) | 2.12 | 11 | 122 |
| SAEM | pynlme `nlmefitsa` (SUND) | 15 | 15 | 148 |
| SAEM | pynlme `nlmefitsa` (SciPy) | 268 | 367 | skipped |
| SAEM | MATLAB `nlmefitsa` (ode15s) | 1739 | 1746 | skipped |
| SAEM | nlmixr2 (rxode2) | 36 | 228 | skipped |

ODE tolerance relative 1e-8, absolute 1e-10 in every program; † marks a fit run at a tighter relative tolerance (shown), because it fails or misses the optimum at the common one.

Log-likelihood each program reports, as a check that the fits are equally good (LME, FOCE and saemix's SAEM; within a method the approximations agree):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 57.58 †1e-10 | 482.68 †1e-10 | 4966.71 †1e-10 |
| LME | pynlme, MATLAB mode (SciPy) | 57.58 | 482.68 | 4966.59 |
| LME | MATLAB `nlmefit` (ode15s) | 57.58 | 481.82 | – |
| LME | pynlme, R mode (SUND) | 57.58 | 482.68 | 4966.68 |
| LME | pynlme, R mode (SciPy) | 57.58 | 482.68 | 4966.70 |
| LME | R `nlme` (deSolve) | 57.58 †1e-12 | 482.68 †1e-12 | – |
| FOCE | pynlme (SUND) | 57.57 †1e-10 | 481.47 †1e-10 | – |
| FOCE | pynlme (SciPy) | 57.58 | – | – |
| FOCE | nlmixr2 (FOCEi, rxode2) | 53.72 | 482.31 | 4890.82 |

Fixed effects: largest difference (log scale) from pynlme's R mode (SciPy) LME fit, as a check that the fits agree (also for SAEM, which reports no log-likelihood by default):

| Method | Program | 10 subjects | 100 subjects | 1000 subjects |
|---|---|---|---|---|
| LME | pynlme, MATLAB mode (SUND) | 0.000 | 0.000 | 0.000 |
| LME | pynlme, MATLAB mode (SciPy) | 0.000 | 0.000 | 0.000 |
| LME | pynlme, R mode (SUND) | 0.000 | 0.000 | 0.000 |
| FOCE | pynlme (SUND) | 0.001 | 0.007 | – |
| FOCE | pynlme (SciPy) | 0.000 | – | – |
| SAEM | pynlme `nlmefitsa` (SUND) | 0.006 | 0.029 | 0.001 |
| SAEM | pynlme `nlmefitsa` (SciPy) | 0.006 | 0.013 | – |

## Findings

At the largest size each program completed:

- **Closed-form model**: for LME pynlme's MATLAB mode is 21 times faster than MATLAB
  and its R mode 1.4 times faster than R `nlme`; for SAEM pynlme is the fastest (2.5
  times faster than MATLAB, `pynlme.r.saemix` 6 times faster than saemix, 85 times
  faster than nlmixr2). For FOCE pynlme is 90 times faster than MATLAB, but nlmixr2's
  FOCEi is 4.7 times faster than pynlme (same optimum).
- **ODE models, LME**: pynlme's R mode is the fastest on all three: 1.6 s on mm at
  1000 subjects (R `nlme` 210 s, MATLAB 47 s), 4.7 s on mmd (`nlme` 697 s, MATLAB 217
  s), 1.1 s on tmdd at 100 (`nlme` 84 s at 1e-12, MATLAB 426 s; 11 s at 1000). The
  MATLAB mode is 11 times faster than MATLAB on mm with SciPy (4.4 s) and 3.7 times
  with SUND (13 s at 1e-10), 430 times on the stiff tmdd model at 100 subjects (SUND,
  0.98 s), but slower than MATLAB on mmd (361 s against 217 s at 1000; SUND at 1e-10,
  likely more outer iterations, as its path is sensitive to small solver
  differences; not investigated).
- **ODE models, SAEM**: pynlme is the fastest from 100 subjects on (mm 56 s at 1000
  against 383 s for nlmixr2; mmd 17 s against 111 s at 100; tmdd 15 s against 228 s
  at 100) and 22 to 118 times faster than MATLAB at 100 subjects. At 10 subjects it
  is close to nlmixr2 (mm 5.8 s against 5.2 s, mmd 17 s against 13 s) and faster on
  tmdd (15 s against 36 s).
- **SAEM estimates**: on tmdd and mmd nlmefitsa's estimates agree with the LME fit
  (fixed effects within 0.03 on the log scale, mmd at 1000 within 0.12). On mm they
  do not: with nlmefitsa's default iterations (150, 150, 100) log Km and log Vmax end
  0.2-0.3 above the LME estimates (true values: log Km 0.41; LME 0.35 and SAEM 0.68
  at 1000 subjects), with SUND and SciPy alike. ka and Km have no random effects,
  and SAEM moves such parameters slowly; with four times the iterations the fit
  reaches the LME estimates (log Km 0.29 at 100 subjects, LME 0.26) and a higher
  importance-sampling log-likelihood (-33.4 against -37.7). MATLAB's nlmefitsa does
  the same on these data (log Km 0.52 with its defaults, 0.33 with four times the
  iterations, at 100 subjects), so it is a property of the default settings, not of
  pynlme or the solver. The SAEM times are for the default settings.
- **ODE models, FOCE**: nlmixr2 is far ahead (14 s on mm, 36 s on mmd, 122 s on tmdd,
  all at 1000 subjects). It computes the model's derivatives from sensitivity
  equations and its outer gradient analytically; pynlme and MATLAB use finite
  differences, which re-solve every subject's conditional modes for each parameter.
  This is pynlme's main gap (SUND sensitivities will allow the same). MATLAB's FOCE
  took 426 s on mm at 10 subjects (pynlme 78 s with SUND at 1e-12, 108 s with SciPy)
  and did not finish tmdd at 10 subjects in 3.3 hours (pynlme 18 s with SUND at 1e-10,
  1053 s with SciPy); we did not investigate why.
- **SUND or SciPy**: with its new defaults (each subject its own integrator, subjects
  in parallel on all 16 threads) SUND is the faster in every comparison except the
  MATLAB mode on mm with many subjects. On the stiff tmdd model it is 5 to 41 times
  faster than SciPy's `ode_model` (LME 11 s against 58 s in the R mode at 1000
  subjects, SAEM 15 s against 367 s at 100). On mm it is 2.5 to 3.2 times faster in
  the R mode (1.6 s against 4.4 s at 1000) and 1.3 to 4 times for SAEM (56 s against
  70 s at 1000, 5.8 s against 23 s at 10). In the MATLAB mode on mm it needs 1e-10
  and is slower at 1000 subjects (13 s against 4.4 s; close at 10 and 100), and FOCE
  needs 1e-12. SciPy solves on one thread, so per core it is the cheaper of the two on
  the non-stiff model. SUND also handles dosing events (mmd).
- **Since SUND 2.5.3** (SUND's coupled batches of 16 subjects on one thread, the
  previous version of this benchmark): LME in the R mode on tmdd at 1000 subjects
  went from 51 s to 11 s, SAEM on tmdd at 10 subjects from 107 s to 15 s, SAEM on mm
  at 1000 from 100 s to 56 s, and the MATLAB mode on mm at 1000 from 75 s to 13 s.
- **mmd at 10 subjects**: MATLAB's pre-fit drives Km to 0, where the model is not
  finite, so MATLAB and pynlme's MATLAB mode both stop there ("MYFUN has returned Inf
  or NaN"); the R mode fits.
- **Fit check**: within a method all programs reach the same fit (log-likelihood
  tables), except MATLAB's FOCE, which ends lower, pynlme's FOCE with SUND on mmd and tmdd
  at 1e-10 (slightly lower, see the ODE solvers above), nlmixr2's FOCEi on tmdd, which
  reports a lower value (its FOCEi likelihood is a different approximation), and
  SAEM on mm with the default iterations (above).

We are not experts in tuning MATLAB, R or nlmixr2: they run as a typical user would
run them, and could likely be made faster.

## Running

```bash
uv run python validation/scaling/make_data.py      # the datasets (data/, not in git)
uv run python validation/scaling/run_pynlme.py     # results/pynlme.json
Rscript validation/scaling/run_r.R                 # results/r.json (nlme, saemix, nlmixr2, deSolve)
# in MATLAB, from the repository root: addpath validation/scaling; run_matlab
uv run python validation/scaling/report.py         # the tables above
# solver accuracy: solver_accuracy.py write, solver_accuracy.R, solver_accuracy.m, solver_accuracy.py
```

`SCALING_SIZES`, `SCALING_MODELS` (R) and `SCALING_METHODS` (MATLAB; pynlme, with
the method names in `results/pynlme.json`, e.g. `SCALING_METHODS="FOCE (SUND)"`)
select parts.
pynlme takes about 70 minutes; MATLAB about three hours without its FOCE on mmd and
tmdd, which takes many hours.
