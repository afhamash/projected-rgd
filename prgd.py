"""Projected Bures--Wasserstein gradient descent.

Companion code for 'Projected Riemannian Gradient Descent for the
Bures--Wasserstein Barycenter: Dimension-Independent Linear Convergence at Unit
Step Size'.

The algorithm is unit-step Riemannian gradient descent on the Bures--Wasserstein
manifold, composed at each iteration with eigenvalue clipping onto the
well-conditioned set ``[alpha I, beta I]``.  Clipping is the exact, non-expansive
projection in the Bures metric, which is what makes the dimension-free rate at
unit step size possible.

Conventions used throughout:

``ensemble``
    ``(n, d, d)`` array holding the positive definite matrices ``R_1, ..., R_n``.
``weights``
    ``(n,)`` probability vector over the ensemble.
``S``
    ``(d, d)`` current iterate.
``alpha``, ``beta``
    spectral bounds of the inputs, defining the well-conditioned set.
``alpha_ref``, ``beta_ref``
    the refined bounds ``alpha'`` and ``beta'`` of the paper.

Everything here is a plain function; nothing is a class.
"""

import hashlib
import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def hermitian_part(matrix):
    """Return the Hermitian part of ``matrix``.

    Used to remove the small asymmetry that accumulates in products of
    numerically computed square roots.
    """
    return (matrix + matrix.conj().T) / 2


def matrix_sqrt(matrix):
    """Return the principal square root of a Hermitian positive semidefinite matrix."""
    values, vectors = np.linalg.eigh(hermitian_part(matrix))
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T


def sqrt_pair(matrix):
    """Return ``(P**0.5, P**-0.5)`` from a single eigendecomposition.

    A gradient step needs both powers, so computing them together halves the
    eigensolver work.
    """
    values, vectors = np.linalg.eigh(hermitian_part(matrix))
    values = np.maximum(values, 0)
    root = (vectors * np.sqrt(values)) @ vectors.conj().T
    inverse_root = (vectors / np.sqrt(values)) @ vectors.conj().T
    return root, inverse_root


def geometric_mean(a, b):
    """Return the matrix geometric mean ``a # b`` of two positive definite matrices.

    This is ``a**0.5 (a**-0.5 b a**-0.5)**0.5 a**0.5``, equivalently the unique
    positive definite solution ``t`` of ``t a**-1 t = b``.
    """
    root, inverse_root = sqrt_pair(a)
    return hermitian_part(root @ matrix_sqrt(inverse_root @ b @ inverse_root) @ root)


def bures_sq(p, q):
    """Return the squared Bures--Wasserstein distance ``B(p, q)``.

    ``B(p, q) = trace(p) + trace(q) - 2 F(p, q)`` with ``F`` the fidelity
    ``trace((p**0.5 q p**0.5)**0.5)``.
    """
    root = matrix_sqrt(p)
    fidelity = np.real(np.trace(matrix_sqrt(root @ q @ root)))
    return float(np.real(np.trace(p) + np.trace(q)) - 2 * fidelity)


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------

def barycenter_functional(S, ensemble, weights):
    """Return the barycenter objective ``f(S) = 0.5 * sum_i w_i B(R_i, S)``."""
    return 0.5 * float(sum(w * bures_sq(R, S) for w, R in zip(weights, ensemble)))


def interval_bounds(ensemble):
    """Return the spectral bounds ``(alpha, beta)`` of the ensemble.

    These are ``min_i lambda_min(R_i)`` and ``max_i lambda_max(R_i)``, so every
    member lies in the well-conditioned set ``[alpha I, beta I]``, and so does
    the barycenter.
    """
    spectra = np.array([np.linalg.eigvalsh(hermitian_part(R)) for R in ensemble])
    return float(spectra.min()), float(spectra.max())


def refined_bounds(ensemble, weights):
    """Return the refined bounds ``(alpha', beta')``.

    ``alpha' = (sum_i w_i lambda_min(R_i)**0.5)**2`` and
    ``beta' = lambda_max(sum_i w_i R_i)``.  They satisfy
    ``alpha <= alpha' <= beta' <= beta`` and still enclose the barycenter, so
    projecting onto ``[alpha' I, beta' I]`` sharpens the rate.
    """
    alpha_ref = float(sum(w * np.sqrt(np.linalg.eigvalsh(hermitian_part(R)).min())
                          for w, R in zip(weights, ensemble)) ** 2)
    average = sum(w * R for w, R in zip(weights, ensemble))
    beta_ref = float(np.linalg.eigvalsh(hermitian_part(average)).max())
    return alpha_ref, beta_ref


def small_step_size(ensemble, weights):
    """Return the small step size ``alpha' / (2 beta')`` of Altschuler et al.

    ``alpha'`` and ``beta'`` are the refined bounds of :func:`refined_bounds`;
    this is the step size at which every small-step comparison in the paper is
    run.
    """
    alpha_ref, beta_ref = refined_bounds(ensemble, weights)
    return alpha_ref / (2 * beta_ref)


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------

def clip(matrix, alpha, beta):
    """Clip the eigenvalues of ``matrix`` into ``[alpha, beta]``.

    By the Projection Lemma of the paper this is the exact Bures projection onto
    the well-conditioned set, and it is non-expansive in the Bures distance.
    """
    values, vectors = np.linalg.eigh(hermitian_part(matrix))
    return (vectors * np.clip(values, alpha, beta)) @ vectors.conj().T


def bw_step(S, ensemble, weights, eta=1.0):
    """Take one Bures--Wasserstein gradient step of size ``eta`` from ``S``.

    The update is ``S' = M S M`` with ``M = (1 - eta) I + eta sum_i w_i (S**-1 # R_i)``.
    At ``eta = 1`` this is the fixed-point map of Alvarez-Esteban et al.
    """
    dimension = S.shape[0]
    root, inverse_root = sqrt_pair(S)
    transported = np.zeros((dimension, dimension), dtype=complex)
    for weight, member in zip(weights, ensemble):
        transported = transported + weight * matrix_sqrt(root @ member @ root)
    step = ((1 - eta) * np.eye(dimension)
            + eta * hermitian_part(inverse_root @ hermitian_part(transported) @ inverse_root))
    return hermitian_part(step @ S @ step)


def bw_map(S, ensemble, weights):
    """Return the unit-step map ``K(S)``, i.e. one gradient step at ``eta = 1``."""
    return bw_step(S, ensemble, weights, eta=1.0)


def projected_bw_gd(ensemble, weights, alpha, beta, S0=None, eta=1.0,
                    n_iters=50, project=True):
    """Run (projected) Bures--Wasserstein gradient descent.

    Returns ``(path, n_clipped)``: the iterates ``S_0, ..., S_T`` and the number
    of steps on which the projection was active.  Pass ``project=False`` for the
    unprojected fixed-point iteration.

    The eigendecomposition of each new iterate is carried into the next step.
    This is the articles's cost claim made concrete: clipping needs that
    decomposition, and the following step needs ``S**0.5`` and ``S**-0.5``, which clipping leaves in the same eigenbasis.  Both variants therefore spend ``n + 1`` eigendecompositions per step.
    """
    dimension = ensemble.shape[-1]
    if S0 is None:
        S = np.eye(dimension, dtype=complex) * (alpha + beta) / 2
    else:
        S = np.array(S0, dtype=complex)
    values, vectors = np.linalg.eigh(hermitian_part(S))
    path, n_clipped = [S], 0

    for _ in range(n_iters):
        adjoint = vectors.conj().T
        root = (vectors * np.sqrt(values)) @ adjoint            # no eigendecomposition
        inverse_root = (vectors / np.sqrt(values)) @ adjoint
        transported = np.zeros((dimension, dimension), dtype=complex)
        for weight, member in zip(weights, ensemble):           # n eigendecompositions
            transported = transported + weight * matrix_sqrt(root @ member @ root)
        step = ((1 - eta) * np.eye(dimension)
                + eta * hermitian_part(inverse_root @ hermitian_part(transported) @ inverse_root))
        # at eta = 1 this collapses to S**-0.5 T**2 S**-0.5, the fixed-point map
        unprojected = hermitian_part(step @ S @ step)

        # one decomposition, serving both the clip and the next step
        values, vectors = np.linalg.eigh(hermitian_part(unprojected))
        if project:
            clipped = np.clip(values, alpha, beta)
            if not np.array_equal(clipped, values):
                n_clipped += 1
            values = clipped
            S = (vectors * values) @ vectors.conj().T
        else:
            S = unprojected
        path.append(S)

    return path, n_clipped


def barycenter(ensemble, weights, n_iters=500):
    """Return the barycenter to high accuracy, for use as a reference optimum."""
    alpha, beta = interval_bounds(ensemble)
    path, _ = projected_bw_gd(ensemble, weights, alpha, beta, n_iters=n_iters)
    return path[-1]


# ---------------------------------------------------------------------------
# Random ensembles
# ---------------------------------------------------------------------------

def make_rng(*parts):
    """Return a generator seeded reproducibly from ``parts``.

    Python's builtin ``hash`` is salted per process, so a stable digest is used
    instead.  Every figure in the paper therefore regenerates bit for bit.
    """
    key = "|".join(repr(part) for part in parts).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)


def random_unitary(d, rng):
    """Return a Haar-random unitary: the unitary polar factor of a Ginibre matrix.

    The polar decomposition of an invertible matrix is unique, so the map
    ``G -> Pol(G)`` is equivariant under left multiplication by unitaries and
    pushes the (left-invariant) Ginibre distribution forward to Haar.
    """
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    w, V = np.linalg.eigh(G.conj().T @ G)
    return G @ ((V / np.sqrt(w)) @ V.conj().T)


def pinned_ensemble(n, dimension, kappa, rng):
    """Return ``n`` matrices with log-uniform spectra pinned to condition number ``kappa``.

    Every member has ``lambda_min = 1`` and ``lambda_max = kappa``, so the
    refined floor is ``alpha' = 1`` identically and the refined condition number
    is comparable across dimensions.  Without pinning, ``min_i lambda_min(R_i)``
    is a minimum of ``n * d`` draws and drifts with the dimension.
    """
    spectra = np.sort(np.exp(rng.uniform(0, np.log(kappa), size=(n, dimension))), axis=1)
    spectra[:, 0], spectra[:, -1] = 1.0, kappa
    ensemble = np.empty((n, dimension, dimension), dtype=complex)
    for index, spectrum in enumerate(spectra):
        unitary = random_unitary(dimension, rng)
        ensemble[index] = (unitary * spectrum) @ unitary.conj().T
    return ensemble


# ---------------------------------------------------------------------------
# Plot style and saving
# ---------------------------------------------------------------------------

def use_style():
    """Apply the matplotlib style shared by every figure in the paper."""
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.rcParams.update({"figure.dpi": 130, "figure.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 11, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5})


def save_figure(figure, name, formats=(".pdf", ".png")):
    """Write ``figure`` to ``name`` in each of ``formats``.

    The paper includes the PDF; the notebook displays the PNG.  Type 42 fonts are
    set in :func:`use_style` so that no Type 3 font reaches the manuscript.
    """
    name = pathlib.Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(name.with_suffix(extension), bbox_inches="tight")
