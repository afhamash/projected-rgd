# ProjectedBWGD

Code related to the article 'Projected Riemannian Gradient Descent for the Bures--Wasserstein Barycenter: Dimension-Independent Linear Convergence at Unit Step Size'.

The notebook `prgd_barycenter.ipynb` reproduces every figure of the paper. It can be run in Google Colab; the first cell clones this repository so that `prgd.py` is importable.

## Files

| file | contents |
|---|---|
| `prgd.py` | the algorithm: the BW gradient step, eigenvalue clipping, the spectral bounds, and the random ensembles |
| `prgdplots.py` | one function per figure of the paper |
| `prgd_barycenter.ipynb` | minimal working examples and the figures |
| `data/` | cached sweeps, so the figures can be redrawn without rerunning the slow experiments |
| `figures/` | the figures as they appear in the paper (`.pdf`) and for viewing (`.png`) |

## What reproduces what

Each figure of the paper is produced by one function, which also prints the
numbers quoted in its caption. Runtimes are for a laptop CPU.

| paper | produced by | runtime | reproduces |
|---|---|---|---|
| Fig. 1 (headline) | `fig_headline()` | ~4 min | (a) median iterations to the floor over 30 pinned d = 2 instances: 16 for the unit step against about 3300 for the small step `alpha'/(2 beta')` (per-instance ratio: median 202, range 141 to 399); (b) the transient exit of Example 13, with undershoots `-2.98e-1` and `-4.43e-3` against the interior limit `+2.52e-3` |
| Fig. 2 (unit vs small) | `fig_unit_vs_small()`, from `data/` | ~10 s cached, **1.6 h** to rerun | median iterations 6, 11, 16 against 175, 1538, 16288 as kappa grows; flat in the dimension |
| Fig. 3 (exit trajectory) | `fig_exit_trajectory()` | ~5 s | both eigenvalues along the trajectory of Example 13: two consecutive iterates with `lambda_min < alpha` |

## Reproducing

Every random ensemble is drawn from a seed derived by `make_rng`, so the figures
regenerate bit for bit. The two sweeps behind Fig. 2 are slow and are shipped in
`data/`; delete the `.npz` files and adapt `fig_unit_vs_small` if you want to
regenerate them from scratch.

## Requirements

`numpy`, `scipy`, `matplotlib`. Install with `pip install -r requirements.txt`.
