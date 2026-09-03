"""Figures for 'Projected Riemannian Gradient Descent for the Bures--Wasserstein
Barycenter'.

Each figure of the paper is produced by one function here; the notebook simply
calls them.  Outputs are written to ``figures/`` as both PDF (used by the
manuscript) and PNG (displayed in the notebook).
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

from prgd import (
    barycenter_functional, bures_sq, bw_map, bw_step, clip, hermitian_part,
    interval_bounds, make_rng, pinned_ensemble, refined_bounds,
    save_figure, small_step_size, use_style,
)

N, KAPPA, SEED = 5, 100.0, 4
SMALL_EVERY = 10         # sample the small step this often when recording it
SMALL_MARK = 200         # ... and mark a donut every this many iterates
UNIT = "#1f5fa8"         # unit-step iterates
SMALL = "#ee8043"        # small-step trajectory


def rotation(angle):
    """Return the 2x2 rotation matrix by ``angle``."""
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


# The certified instance of the paper: an attacker with a wide spectrum, two mild
# targets, a strictly interior optimum and a strictly interior start.
HARD_INSTANCE = dict(
    angles=(0.246, 0.338, 0.253),
    spectra=[(1.05, 315.0), (1.0501, 1.575), (1.0502, 4.41)],
    weights=(0.12, 0.827, 0.053),
    start=(1.473, (1.11, 310.0)),
)
GRID_COLOUR = "#e3e2dd"
INK_MUTED = "#8a8880"
INK_SECONDARY = "#52514e"
PROJECTED_COLOUR = "#2a78d6"
VIOLATION_COLOUR = "#e34948"
SMALL_STEP_COLOUR = "#8a8880"


def rotated(angle, eigenvalues):
    """Return ``Q diag(eigenvalues) Q^T`` with ``Q`` the rotation by ``angle``."""
    return (rotation(angle) @ np.diag(eigenvalues) @ rotation(angle).T).astype(complex)


def hard_instance():
    """Return ``(ensemble, weights, S0, alpha, beta)`` for the certified instance.

    The ensemble has condition number ``kappa = 300``, the start is strictly
    inside the well-conditioned set, and so is the barycenter -- yet the
    unprojected unit step leaves the set for two consecutive iterations.
    """
    ensemble = np.array([rotated(angle, spectrum) for angle, spectrum
                         in zip(HARD_INSTANCE["angles"], HARD_INSTANCE["spectra"])])
    weights = np.array(HARD_INSTANCE["weights"])
    S0 = rotated(*HARD_INSTANCE["start"])
    alpha, beta = interval_bounds(ensemble)
    return ensemble, weights, S0, alpha, beta



# ---------------------------------------------------------------------------
# Figure 1: the headline figure -- convergence bands and the transient exit
# ---------------------------------------------------------------------------

HEADLINE_FLOOR = 1e-12   # the double-precision cancellation floor of the gaps
LIGHTBLUE = "#6baed6"    # lambda_min in panel (b)
VIOLATION_RED = "#d62728"


def convergence_data(seed=SEED):
    """Return normalised unit and small-step gap curves on one instance.

    The instance family is the pinned ensemble at ``d = 2``: every member has
    spectrum exactly ``{1, kappa}`` in a Haar-random eigenbasis.  The small
    step is ``eta = alpha'/(2 beta')`` of Altschuler et al., and gaps are
    normalised by the initial unit-step gap.
    """
    rng = make_rng("headline", 2, N, KAPPA, seed)
    ensemble = pinned_ensemble(N, 2, KAPPA, rng)
    weights = np.full(N, 1.0 / N)
    alpha, beta = interval_bounds(ensemble)
    eta = small_step_size(ensemble, weights)

    S0 = np.eye(2, dtype=complex) * (alpha + beta) / 2.0
    S, unit = S0.copy(), [S0]
    for _ in range(60):
        K = bw_map(S, ensemble, weights)
        step = bures_sq(S, K)
        S = clip(K, alpha, beta)
        unit.append(S)
        if step <= 1e-26:
            break
    S_star = unit[-1]

    S, small, small_ts = S0.copy(), [S0], [0]
    for t in range(1, 60_001):
        S = bw_step(S, ensemble, weights, eta=eta)
        if t % SMALL_EVERY == 0:
            small.append(S)
            small_ts.append(t)
        if np.linalg.norm(S - S_star) < 1e-9:
            break

    reference = S_star.copy()
    for _ in range(200):
        reference = clip(bw_map(reference, ensemble, weights), alpha, beta)
    f_star = barycenter_functional(reference, ensemble, weights)
    gap_unit = np.array([barycenter_functional(S, ensemble, weights) - f_star
                         for S in unit])
    gap_small = np.array([barycenter_functional(S, ensemble, weights) - f_star
                          for S in small])
    return gap_unit / gap_unit[0], gap_small / gap_unit[0], np.array(small_ts, dtype=float)


def convergence_ensemble(seeds=range(30)):
    """Return the gap curves of every instance of the family, for the bands."""
    runs = []
    for seed in seeds:
        gap_unit, gap_small, small_ts = convergence_data(seed)
        runs.append((gap_unit, gap_small, small_ts))
    return runs


def median_curve(curves, floor=HEADLINE_FLOOR):
    """Return the pointwise median of ``(t, gap)`` curves on their common grid."""
    grid = np.unique(np.concatenate([t for t, _ in curves]))
    median = np.median([np.interp(grid, t, np.maximum(g, floor)) for t, g in curves],
                       axis=0)
    return grid, median


def draw_convergence(axis, runs):
    """Panel (a): translucent per-instance curves with opaque medians on top."""
    unit_curves = [(np.arange(len(gu), dtype=float), gu) for gu, _, _ in runs]
    small_curves = [(ts, gs) for _, gs, ts in runs]
    for t, g in unit_curves:
        keep = g > HEADLINE_FLOOR
        axis.plot(np.maximum(t[keep], 0.7), g[keep], ":", color=UNIT, alpha=0.30,
                  lw=0.9, zorder=2)
    for t, g in small_curves:
        keep = g > HEADLINE_FLOOR
        axis.plot(np.maximum(t[keep], 0.7), g[keep], ":", color=SMALL, alpha=0.30,
                  lw=0.9, zorder=2)

    tu, mu = median_curve(unit_curves)
    keep_u = mu > HEADLINE_FLOOR
    axis.plot(np.maximum(tu[keep_u], 0.7), mu[keep_u], "-", color=UNIT, lw=2.2, zorder=4)
    axis.plot(np.maximum(tu[keep_u], 0.7), mu[keep_u], "o", mfc="none", mec=UNIT,
              mew=1.4, ms=4.6, linestyle="none", zorder=5,
              label=r"unit step RGD ($\eta = 1$)")
    ts, ms = median_curve(small_curves)
    keep_s = ms > HEADLINE_FLOOR
    axis.plot(np.maximum(ts[keep_s], 0.7), ms[keep_s], "-", color=SMALL, lw=2.2, zorder=4)
    marks = keep_s & (np.mod(ts, SMALL_MARK) == 0) & (ts > 0)
    axis.plot(ts[marks], ms[marks], "o", mfc="none", mec=SMALL, mew=1.4, ms=4.6,
              linestyle="none", zorder=5,
              label=r"small step RGD ($\eta = \alpha'/2\beta'$)")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.6, 2.0 * ts[keep_s].max())
    axis.set_ylim(HEADLINE_FLOOR * 0.5, 3.0)
    axis.set_xlabel("iteration $t$")
    axis.set_ylabel(r"$[f(S_t) - f(S_\star)]\,/\,[f(S_0) - f(S_\star)]$", fontsize=9)
    axis.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    axis.set_title(r"(a) unit against small step   ($d = 2$, $n = 5$, $\kappa = 10^2$)",
                   fontsize=10)
    return int(tu[keep_u].max()), int(ts[keep_s].max())


def draw_exit(axis, spectra, alpha, beta, dev_star):
    """Panel (b): the deviation view of the exit instance, with a full-picture inset."""
    t = np.arange(len(spectra))
    lam_min, lam_max = spectra[:, 0], spectra[:, -1]
    violating = lam_min < alpha * (1.0 - 1e-9)
    dev = lam_min - alpha

    axis.axhline(0.0, color=INK_SECONDARY, lw=1.0, ls=":", zorder=1)
    axis.axhline(dev_star, color=INK_SECONDARY, lw=1.0, ls="-.", zorder=1)
    axis.annotate(r"$\lambda_{\min}(S_\star) - \alpha$", (0.6, dev_star),
                  textcoords="offset points", xytext=(0, 5), fontsize=8.5,
                  color=INK_SECONDARY)
    axis.annotate(r"$\lambda_{\min} = \alpha$", (t[-1] + 0.85, 0.0), fontsize=9,
                  ha="right", va="bottom")
    axis.axhspan(-1.0, 0.0, color=VIOLATION_RED, alpha=0.10, lw=0, zorder=0)
    axis.plot(t, dev, "o", ls="--", color=LIGHTBLUE, ms=5.0, lw=1.3, zorder=3)
    axis.plot(t[violating], dev[violating], "o", color=VIOLATION_RED, ms=8.0,
              mfc="none", mew=1.8, zorder=5)
    for i in (1, 2):
        axis.annotate(f"{dev[i]:.2e}", (t[i], dev[i]), textcoords="offset points",
                      xytext=(10, -3), fontsize=8, color=VIOLATION_RED)
    axis.set_yscale("symlog", linthresh=1e-4)
    axis.set_ylim(-1.0, 0.3)
    axis.set_xlim(-0.4, t[-1] + 0.9)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("iteration $t$")
    axis.set_ylabel(r"$\lambda_{\min}(S_t) - \alpha$")
    axis.set_title(r"(b) transient exit from $[\alpha I, \beta I]$   ($\kappa = 300$)",
                   fontsize=10)

    inset = axis.inset_axes([0.55, 0.07, 0.42, 0.38])
    inset.axhspan(alpha, beta, color="#d8d7d2", alpha=0.55, lw=0, zorder=0)
    inset.axhline(alpha, color=INK_SECONDARY, lw=0.8, ls=":", zorder=1)
    inset.axhline(beta, color=INK_SECONDARY, lw=0.8, ls=":", zorder=1)
    inset.plot(t, lam_max, "-o", color=UNIT, ms=2.8, lw=1.0, zorder=3)
    inset.plot(t, lam_min, "o", ls="--", color=LIGHTBLUE, ms=2.8, lw=1.0, zorder=3)
    inset.plot(t[violating], lam_min[violating], "o", color=VIOLATION_RED, ms=5.0,
               mfc="none", mew=1.3, zorder=5)
    inset.set_yscale("log")
    inset.annotate(r"$\beta$", (t[-1] + 0.3, beta), fontsize=8, va="center")
    inset.annotate(r"$\alpha$", (t[-1] + 0.3, alpha), fontsize=8, va="center")
    inset.text(0.35, 0.60, r"$\lambda_{\max}(S_t)$", transform=inset.transAxes,
               fontsize=7.5, color=UNIT)
    inset.text(0.35, 0.13, r"$\lambda_{\min}(S_t)$", transform=inset.transAxes,
               fontsize=7.5, color=LIGHTBLUE)
    inset.set_xlim(-0.4, t[-1] + 1.6)
    inset.tick_params(labelsize=6.5)
    inset.xaxis.set_major_locator(MaxNLocator(integer=True))

###################

def fig_headline(seeds=range(30), n_iters=12, outdir=None, quiet=False):
    """Draw Figure 1: convergence bands with medians, and the transient exit.

    Panel (a): the optimality gap of unit-step against small-step RGD on all
    thirty instances of the pinned family, translucent per instance, opaque at
    the median.  Panel (b): the certified exit instance, as the deviation
    ``lambda_min(S_t) - alpha``, with the interior limit of the unprojected
    iteration marked and the full two-eigenvalue picture in the inset.
    """
    results = Path(outdir) if outdir is not None else Path("figures")
    plt.close("all")
    plt.style.use("ggplot")
    plt.rcParams.update({"figure.dpi": 130, "figure.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 10, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "axes.titlesize": 10})

    runs = convergence_ensemble(seeds)

    ensemble, weights, S0, alpha, beta = hard_instance()
    S = np.array(S0, dtype=complex)
    spectra = []
    for _ in range(n_iters + 1):
        spectra.append(np.linalg.eigvalsh(hermitian_part(S)))
        S = bw_map(S, ensemble, weights)
    spectra = np.array(spectra)

    S = np.array(S0, dtype=complex)
    for _ in range(3000):
        S = bw_map(S, ensemble, weights)
    dev_star = float(np.linalg.eigvalsh(hermitian_part(S)).min()) - alpha

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    n_unit, n_small = draw_convergence(ax_a, runs)
    draw_exit(ax_b, spectra, alpha, beta, dev_star)
    figure.tight_layout()
    save_figure(figure, results / "fig_headline.png")

    if not quiet:
        ratios = []
        for gap_unit, gap_small, small_ts in runs:
            hit_u = np.nonzero(gap_unit <= HEADLINE_FLOOR)[0]
            hit_s = np.nonzero(gap_small <= HEADLINE_FLOOR)[0]
            if hit_u.size and hit_s.size:
                ratios.append(small_ts[hit_s[0]] / hit_u[0])
        ratios = np.array(ratios)
        dev = spectra[:, 0] - alpha
        print(f"(a) medians over {len(runs)} instances: unit reaches the floor at "
              f"t = {n_unit}, small at t = {n_small} ({n_small / n_unit:.0f}x)")
        print(f"    per-instance iteration ratio at the floor: median "
              f"{np.median(ratios):.0f}, range {ratios.min():.0f} to {ratios.max():.0f}")
        print(f"(b) lambda_min - alpha at t = 1, 2, 3: "
              f"{dev[1]:.3e}, {dev[2]:.3e}, {dev[3]:.3e}; limit {dev_star:.3e}")
        print(f"wrote {results}/fig_headline.pdf and .png")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2: unit against small step, graded by conditioning and by dimension
# ---------------------------------------------------------------------------

UNIT_SHADES = ["#a8c8ee", "#5b9bdd", "#1f5fa8"]
SMALL_SHADES = ["#f6b48c", "#ee8043", "#b8500f"]
BOUND_SHADES = ["#a1d99b", "#41ab5d", "#00682c"]
INK_PRIMARY = "#2f2e2b"
FLOOR = 1e-13


def pinned_ensemble_ported(d, n, kappa, rng):
    """Adapter: the sweep scripts call with (d, n, kappa), prgd uses (n, d, kappa)."""
    return pinned_ensemble(n, d, kappa, rng)


def median_refined_ratio(tag, d, n, kappa, n_seeds):
    """Median alpha'/beta' over the instances of one band (same seeds as the sweep)."""
    vals = []
    for s in range(n_seeds):
        rng = make_rng(tag, d, n, kappa, s)
        ens = pinned_ensemble_ported(d, n, kappa, rng)
        w = np.full(n, 1.0 / n)
        a_ref, _ = refined_bounds(ens, w)
        b_ref = float(np.max(np.linalg.eigvalsh(sum(wi * R for wi, R in zip(w, ens)))))
        vals.append(a_ref / b_ref)
    return float(np.median(vals))


def guarantee_curve(ax, ratio, shade, label, handles):
    """Dotted guarantee (1 - ratio^{3/2})^t, drawn down to the plotting floor."""
    rate = ratio ** 1.5
    t_end = int(np.ceil(np.log(FLOOR) / np.log1p(-rate)))
    t = np.unique(np.round(np.geomspace(1, t_end, 400)).astype(int))
    handles += ax.plot(t, (1.0 - rate) ** t, ":", color=shade, lw=2.0, zorder=5,
                       dashes=(0.7, 2.2), dash_capstyle="round", label=label)


class _Blank:
    """Marker for a text-only legend cell (row label or column title)."""


class _HandlerBlank(HandlerBase):
    """Collapses the handle box so text-only cells start at the column edge."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.set_width(0)
        return Line2D([], [], linestyle="none")


def matrix_legend(ax, row_labels, cols, loc="center right", fontsize=6.4):
    """Legend as a grid: one row per shade, one column per series type.

    Avoids repeating the parameter value in every entry. matplotlib fills a
    multi-column legend column-major, so entries are emitted column by column:
    a leading column of row labels, then one column of handles per series.
    Text-only cells use a zero-width handle box, so the column titles and the
    row labels are flush with the left edge of their column.
    """
    handles, labels = [_Blank()], [""]                # corner cell
    for lbl in row_labels:                            # column 0: row labels
        handles.append(_Blank()); labels.append(lbl)
    for title, hs in cols:                            # one column per series
        handles.append(_Blank()); labels.append(title)
        for h in hs:
            handles.append(h); labels.append("")
    return ax.legend(handles, labels, loc=loc, ncol=1 + len(cols),
                     fontsize=fontsize, frameon=True, facecolor="white",
                     edgecolor="none", framealpha=0.93, handlelength=4.6,
                     handletextpad=0.0, columnspacing=1.6, labelspacing=0.45,
                     borderpad=0.5, handler_map={_Blank: _HandlerBlank()})


def instance_band(ax, trajs, shade, label, handles):
    """Per-instance trajectories dotted and translucent; opaque solid median."""
    for t, g in trajs:
        ax.plot(np.maximum(t, 0.7), np.maximum(g, FLOOR), ":", color=shade,
                alpha=0.35, lw=0.9, zorder=2)
    grid = np.unique(np.concatenate([t for t, _ in trajs]))
    med = np.median([np.interp(grid, t, np.maximum(g, FLOOR)) for t, g in trajs],
                    axis=0)
    handles += ax.plot(np.maximum(grid, 0.7), med, "-", color=shade, lw=2.4,
                       zorder=4, label=label)


def fig_unit_vs_small(outdir=None, quiet=False):
    """Draw Figure 2: unit against small step, graded by kappa and by dimension.

    Reads the cached sweeps in ``data/``.  Recomputing them takes about six
    minutes for panel (a) and over an hour for panel (b), so they are shipped.
    """
    results = Path(outdir) if outdir is not None else Path("figures")
    plt.close("all")
    ka = np.load(Path("data") / "data_unit_vs_small.npz")
    di = np.load(Path("data") / "data_unit_vs_small_dim.npz")
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 130,
        "figure.facecolor": "white",
        # TrueType rather than Type 3, as required for arXiv/journal PDFs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)

    # (a) kappa gradation, d fixed.
    hu, hs, hb = [], [], []
    for i, kappa in enumerate(ka["kappas"]):
        key = f"{kappa:.0f}"
        n = int(ka["n_seeds"])
        instance_band(ax_a, [(np.arange(len(ka[f"unit_{key}_{s}"])),
                     ka[f"unit_{key}_{s}"] / ka[f"unit_{key}_{s}"][0])
                    for s in range(n)], UNIT_SHADES[i],
             rf"unit step, $\kappa = 10^{{{int(np.log10(kappa))}}}$", hu)
        instance_band(ax_a, [(ka[f"small_t_{key}_{s}"],
                     ka[f"small_g_{key}_{s}"] / ka[f"small_g_{key}_{s}"][0])
                    for s in range(n)], SMALL_SHADES[i],
             rf"small step, $\kappa = 10^{{{int(np.log10(kappa))}}}$", hs)
        ratio = median_refined_ratio("unit-vs-small-pinned", int(ka["d"]), int(ka["n"]),
                                     float(kappa), n)
        guarantee_curve(ax_a, ratio, BOUND_SHADES[i],
              rf"$\kappa'^{{3/2}}$ bound, $\kappa = 10^{{{int(np.log10(kappa))}}}$", hb)
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_ylim(FLOOR * 0.5, 4.0)
    ax_a.set_xlabel("iteration $t$")
    ax_a.set_ylabel(r"$[f(S_t) - f(S_\star)] \,/\, [f(S_0) - f(S_\star)]$")
    ax_a.set_title(rf"(a) varying $\kappa$   ($d = {int(ka['d'])}$, $n = {int(ka['n'])}$)",
                   fontsize=9.5, color=INK_PRIMARY)
    matrix_legend(ax_a,
                  [rf"$\kappa = 10^{{{int(np.log10(k))}}}$" for k in ka["kappas"]],
                  [("unit step", hu), ("small step", hs),
                   (r"$\kappa'^{3/2}$ bound", hb)])

    # (b) dimension gradation, kappa fixed.
    hu, hs, hb = [], [], []
    for i, d in enumerate(di["dims"]):
        key = f"{d}"
        n = int(di["n_seeds"])
        instance_band(ax_b, [(np.arange(len(di[f"unit_{key}_{s}"])), di[f"unit_{key}_{s}"])
                    for s in range(n)], UNIT_SHADES[i], rf"unit step, $d = {d}$", hu)
        instance_band(ax_b, [(di[f"small_t_{key}_{s}"], di[f"small_g_{key}_{s}"])
                    for s in range(n)], SMALL_SHADES[i], rf"small step, $d = {d}$", hs)
        ratio = median_refined_ratio("unit-vs-small-dim-pinned", int(d), int(di["n"]),
                                     float(di["kappa"]), n)
        guarantee_curve(ax_b, ratio, BOUND_SHADES[i],
              rf"$\kappa'^{{3/2}}$ bound, $d = {d}$", hb)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlabel("iteration $t$")
    ax_b.set_title(rf"(b) varying $d$   ($n = {int(di['n'])}$, "
                   rf"$\kappa = {float(di['kappa']):g}$)", fontsize=9.5,
                   color=INK_PRIMARY)
    matrix_legend(ax_b, [rf"$d = {d}$" for d in di["dims"]],
                  [("unit step", hu), ("small step", hs),
                   (r"$\kappa'^{3/2}$ bound", hb)])

    fig.tight_layout()
    save_figure(fig, results / "fig_unit_vs_small.png")
    if not quiet:
        print(f"wrote {results}/fig_unit_vs_small.pdf and .png")
    plt.show()




# ---------------------------------------------------------------------------
# Figure 3: the eigenvalue trajectory of the exit instance
# ---------------------------------------------------------------------------


def fig_exit_trajectory(n_iters=12, outdir=None, quiet=False):
    """Draw Figure 3: both eigenvalues along the unprojected unit-step trajectory.

    The feasible band ``[alpha, beta]`` is shaded and the infeasible iterates are
    marked; the inset resolves the two undershoots against the certified interior
    floor of the optimum.
    """
    results = Path(outdir) if outdir is not None else Path("figures")
    plt.close("all")
    ensemble, weights, S0, alpha, beta = hard_instance()

    S = np.array(S0, dtype=complex)
    spectra = []
    for _ in range(n_iters + 1):
        spectra.append(np.linalg.eigvalsh(hermitian_part(S)))
        S = bw_map(S, ensemble, weights)
    spectra = np.array(spectra)

    draw_exit_trajectory(spectra, alpha, beta, results / "fig_exit_trajectory.png")
    if not quiet:
        below = [int(t) for t, lam in enumerate(spectra[:, 0]) if lam < alpha * (1 - 1e-9)]
        print(f"iterates with lambda_min < alpha: t = {below}")
        print(f"wrote {results}/fig_exit_trajectory.pdf and .png")


def draw_exit_trajectory(spectra, alpha, beta, path, paper=True):
    """Draw the eigenvalue trajectory with the feasible band shaded and a zoom inset."""
    use_style()
    t = np.arange(len(spectra))
    lam_min, lam_max = spectra[:, 0], spectra[:, -1]
    # A strict violation, guarded against the floating-point dust that makes the
    # feasible starting iterate register as marginally infeasible.
    violating = lam_min < alpha * (1.0 - 1e-9)

    figure, ax = plt.subplots(figsize=(6.4, 4.0))

    ax.axhspan(alpha, beta, color=GRID_COLOUR, alpha=0.6, lw=0, zorder=0,
               label=r"feasible band $[\alpha I, \beta I]$")
    # Labelled on the left, so the legend can sit outside the axes on the right.
    for level, name in ((beta, r"$\beta$"), (alpha, r"$\alpha$")):
        ax.axhline(level, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
        ax.annotate(name, xy=(t[0], level), xytext=(4, 4),
                    textcoords="offset points", fontsize=9,
                    color=INK_SECONDARY, va="bottom")

    ax.plot(t, lam_max, "-o", color=PROJECTED_COLOUR, zorder=3,
            label=r"$\lambda_{\max}(S_t)$")
    ax.plot(t, lam_min, "-o", color='#eb6834', zorder=3,
            label=r"$\lambda_{\min}(S_t)$")
    ax.plot(t[violating], lam_min[violating], "o", color=VIOLATION_COLOUR,
            markersize=7, zorder=4, label=r"$\lambda_{\min} < \alpha$")

    ax.set_yscale("log")
    ax.set_xlim(-0.3, t[-1] + 0.3)
    ax.set_ylim(min(alpha * 0.82, lam_min.min() * 0.88), beta * 1.35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("eigenvalues of $S_t$")
    n_out = int(violating.sum())
    # Paper figures carry no in-figure title: the caption does that job.
    if not paper:
        ax.set_title(f"Unit-step BW-GD leaves the well-conditioned set for "
                     f"{n_out} step{'s' if n_out != 1 else ''}, then returns")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    add_zoom_inset(ax, t, lam_min, lam_max, alpha, beta, violating)

    save_figure(figure, path.with_suffix(".png"))
    plt.show()


def inset_band(ax, lam_min, lam_max) :
    """Vertical extent, in axes fractions, of the empty band between the two curves.

    The curves separate by orders of magnitude, and by different amounts in each
    instance, so the inset is positioned from the data rather than pinned to
    fixed coordinates where it would land on a line.
    """
    lo, hi = ax.get_ylim()
    span = np.log(hi / lo)

    def frac(y: float) -> float:
        return float(np.log(y / lo) / span)

    pad = 0.06
    return frac(lam_min.max()) + pad, frac(lam_max.min()) - pad


def add_zoom_inset(ax, t, lam_min, lam_max, alpha, beta, violating) :
    """Close-up of the excursion, as the signed deviation ``lambda_min - alpha``.

    The deviations span several orders of magnitude when the excursion lasts more
    than one step, so the axis is symmetric-log: it resolves a shallow second
    undershoot alongside a deep first one, and keeps the sign change visible.
    """
    bottom, top = inset_band(ax, lam_min, lam_max)
    # If the curves crowd together, fall back to the widest band available.
    if top - bottom < 0.30:
        bottom, top = (0.55, 0.90) if bottom < 0.5 else (0.10, 0.45)
    axin = ax.inset_axes((0.32, bottom, 0.44, top - bottom))
    dev = lam_min - alpha
    # The linear zone must clear the deviation dust at a marginally feasible S_0
    # (where lambda_min equals alpha to machine precision), or the symlog axis
    # stretches over fifteen empty decades.
    significant = np.abs(dev)[np.abs(dev) > 1e-10 * alpha]
    linthresh = significant.min() * 0.5 if significant.size else np.abs(dev).max() * 1e-3
    limit = np.abs(dev).max() * 4.0

    axin.axhspan(0.0, limit, color=GRID_COLOUR, alpha=0.6, lw=0, zorder=0)
    axin.axhspan(-limit, 0.0, color=VIOLATION_COLOUR, alpha=0.10, lw=0, zorder=0)
    axin.axhline(0.0, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
    axin.plot(t, dev, "-o", color='#eb6834', markersize=4, zorder=3)
    axin.plot(t[violating], dev[violating], "o", color=VIOLATION_COLOUR,
              markersize=5.5, zorder=4)

    for t_bad in t[violating]:
        axin.annotate(f"{dev[t_bad]:.2e}", xy=(t_bad, dev[t_bad]), xytext=(7, 0),
                      textcoords="offset points", fontsize=7, color=VIOLATION_COLOUR,
                      va="center")

    axin.set_yscale("symlog", linthresh=linthresh)
    # Every second decade: a full decade ladder is unreadable at inset size.
    axin.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=linthresh, base=100))
    axin.set_xlim(-0.3, min(t[-1], int(violating.sum()) + 4) + 0.3)
    axin.set_ylim(-limit, limit)
    axin.xaxis.set_major_locator(MaxNLocator(integer=True))
    axin.tick_params(labelsize=7)
    axin.set_title(r"$\lambda_{\min}(S_t) - \alpha$, symlog", fontsize=8,
                   color=INK_SECONDARY, pad=3)
