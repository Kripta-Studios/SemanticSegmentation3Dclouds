from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

TW_FEATURE_NAMES = [
    "linear_eigenvalue_1",
    "linear_eigenvalue_2",
    "linear_eigenvalue_3",
    "linearity",
    "planarity",
    "scattering",
    "normal_z_abs",
    "quadratic_deviation",
    "quadric_eigensum",
    "quadric_eigenentropy",
    "gradient_norm",
    "laplacian",
    "hessian_frobenius",
    "hessian_abs_det",
    "hessian_saddleness",
    "kappa_min",
    "kappa_max",
    "mean_curvature",
    "gaussian_curvature",
    "curvature_abs_mean",
    "curvature_abs_max",
    "curvedness",
    "shape_index",
    "quadratic_valid",
    "tw_valid",
]

TW_FEATURE_DIM = len(TW_FEATURE_NAMES)


def _monomials(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.stack(
        [x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, np.ones_like(x)],
        axis=-1,
    )


def _jacobian_monomials(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    jac = np.zeros((*x.shape, 3, 10), dtype=np.float64)
    jac[..., 0, 0] = 2 * x
    jac[..., 0, 3] = y
    jac[..., 0, 4] = z
    jac[..., 0, 6] = 1.0
    jac[..., 1, 1] = 2 * y
    jac[..., 1, 3] = x
    jac[..., 1, 5] = z
    jac[..., 1, 7] = 1.0
    jac[..., 2, 2] = 2 * z
    jac[..., 2, 4] = x
    jac[..., 2, 5] = y
    jac[..., 2, 8] = 1.0
    return jac


def _safe_entropy(values: np.ndarray, axis: int = 1) -> np.ndarray:
    vals = np.maximum(values, 0.0)
    denom = np.sum(vals, axis=axis, keepdims=True)
    probs = np.divide(vals, denom, out=np.zeros_like(vals), where=denom > 0)
    terms = np.zeros_like(probs)
    mask = probs > 0
    terms[mask] = probs[mask] * np.log(probs[mask])
    return -np.sum(terms, axis=axis)


def _tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.zeros_like(normals)
    smallest = np.argmin(np.abs(normals), axis=1)
    helper[np.arange(normals.shape[0]), smallest] = 1.0
    alpha = np.cross(normals, helper)
    alpha /= np.maximum(np.linalg.norm(alpha, axis=1, keepdims=True), 1e-12)
    beta = np.cross(normals, alpha)
    beta /= np.maximum(np.linalg.norm(beta, axis=1, keepdims=True), 1e-12)
    return alpha, beta


def _batch_rank(matrix: np.ndarray, rel_tol: float = 1e-8) -> np.ndarray:
    singular = np.linalg.svd(matrix, compute_uv=False)
    limit = rel_tol * np.maximum(singular[:, :1], 1.0)
    return np.sum(singular > limit, axis=1)


def _process_weingarten_batch(
    centered: np.ndarray,
    eigenthreshold: float = 1e-5,
    tikhonov: float = 1e-6,
) -> np.ndarray:
    centered = centered.astype(np.float64, copy=False)
    batch, k_pts, _ = centered.shape
    out = np.zeros((batch, TW_FEATURE_DIM), dtype=np.float64)
    if k_pts < 3:
        return out.astype(np.float32)

    x, y, z = centered[:, :, 0], centered[:, :, 1], centered[:, :, 2]

    cov = np.matmul(centered.transpose(0, 2, 1), centered) / max(k_pts - 1, 1)
    lin_eval, lin_evec = np.linalg.eigh(cov)
    lin_eval_desc = lin_eval[:, ::-1]
    normal = lin_evec[:, :, 0]
    linear_rank = np.sum(lin_eval > eigenthreshold, axis=1)
    valid_linear = linear_rank >= 2
    denom = np.maximum(lin_eval_desc[:, 0], 1e-12)
    out[:, 0:3] = lin_eval_desc
    out[:, 3] = (lin_eval_desc[:, 0] - lin_eval_desc[:, 1]) / denom
    out[:, 4] = (lin_eval_desc[:, 1] - lin_eval_desc[:, 2]) / denom
    out[:, 5] = lin_eval_desc[:, 2] / denom
    out[:, 6] = np.abs(normal[:, 2])

    T = _monomials(x, y, z)
    quad_rank = _batch_rank(T)
    surface_variation = lin_eval[:, 0] / np.maximum(np.sum(lin_eval, axis=1), 1e-12)
    nondegenerate = valid_linear & (quad_rank >= 9) & (surface_variation > eigenthreshold)
    out[:, -1] = valid_linear.astype(np.float64)
    if not np.any(nondegenerate):
        return out.astype(np.float32)

    jac = _jacobian_monomials(x, y, z)
    M = np.matmul(T.transpose(0, 2, 1), T) / k_pts
    N = np.einsum("bksi,bksj->bij", jac, jac) / k_pts
    trace_n = np.trace(N, axis1=1, axis2=2)
    N_reg = N + (tikhonov * np.maximum(trace_n, 1e-12))[:, None, None] * np.eye(10)[None, :, :]

    quad_eval = np.zeros((batch, 10), dtype=np.float64)
    quad_evec = np.zeros((batch, 10, 10), dtype=np.float64)
    solved = np.zeros(batch, dtype=bool)
    for i in np.flatnonzero(nondegenerate):
        try:
            L = np.linalg.cholesky(N_reg[i])
            L_inv = np.linalg.inv(L)
            sym = L_inv @ M[i] @ L_inv.T
            vals, vecs_prime = np.linalg.eigh(sym)
            vecs = L_inv.T @ vecs_prime
            quad_eval[i] = vals
            quad_evec[i] = vecs
            solved[i] = True
        except np.linalg.LinAlgError:
            continue

    good = solved & np.isfinite(quad_eval).all(axis=1)
    if not np.any(good):
        return out.astype(np.float32)

    coeffs = np.zeros((batch, 10), dtype=np.float64)
    for i in np.flatnonzero(good):
        eig_idx = int(np.argmin(np.abs(quad_eval[i])))
        coeffs[i] = quad_evec[i, :, eig_idx]
        norm = np.linalg.norm(coeffs[i])
        if norm > 0:
            coeffs[i] /= norm

    A, B, C = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
    D, E, F = coeffs[:, 3], coeffs[:, 4], coeffs[:, 5]
    G, H, I = coeffs[:, 6], coeffs[:, 7], coeffs[:, 8]
    gradient = np.stack([G, H, I], axis=1)
    gradient_norm = np.linalg.norm(gradient, axis=1)
    nonsingular = good & (gradient_norm > 1e-10)
    out[:, 7] = np.maximum(quad_eval[:, 0], 0.0)
    qabs = np.abs(quad_eval)
    out[:, 8] = np.sum(qabs, axis=1)
    out[:, 9] = _safe_entropy(qabs, axis=1)
    out[:, 10] = gradient_norm

    hessian = np.zeros((batch, 3, 3), dtype=np.float64)
    hessian[:, 0, 0] = 2 * A
    hessian[:, 1, 1] = 2 * B
    hessian[:, 2, 2] = 2 * C
    hessian[:, 0, 1] = hessian[:, 1, 0] = D
    hessian[:, 0, 2] = hessian[:, 2, 0] = E
    hessian[:, 1, 2] = hessian[:, 2, 1] = F
    out[:, 11] = np.trace(hessian, axis1=1, axis2=2)
    out[:, 12] = np.linalg.norm(hessian, axis=(1, 2))
    out[:, 13] = np.abs(np.linalg.det(hessian))
    h_eval = np.linalg.eigvalsh(hessian)
    h_min, h_max = h_eval[:, 0], h_eval[:, -1]
    h_low = np.where(np.abs(h_min) <= np.abs(h_max), h_min, h_max)
    h_high = np.where(np.abs(h_min) > np.abs(h_max), h_min, h_max)
    out[:, 14] = -np.divide(h_low, h_high, out=np.zeros_like(h_low), where=np.abs(h_high) > 1e-12)
    out[:, 23] = good.astype(np.float64)

    if np.any(nonsingular):
        idx = np.flatnonzero(nonsingular)
        normals = gradient[idx] / gradient_norm[idx, None]
        alpha, beta = _tangent_basis(normals)
        Hn = hessian[idx]
        aa = np.einsum("bi,bij,bj->b", alpha, Hn, alpha)
        ab = np.einsum("bi,bij,bj->b", alpha, Hn, beta)
        bb = np.einsum("bi,bij,bj->b", beta, Hn, beta)
        W = np.zeros((idx.size, 2, 2), dtype=np.float64)
        W[:, 0, 0] = -aa / gradient_norm[idx]
        W[:, 0, 1] = -ab / gradient_norm[idx]
        W[:, 1, 0] = W[:, 0, 1]
        W[:, 1, 1] = -bb / gradient_norm[idx]
        curv = np.linalg.eigvalsh(W)
        k_min = curv[:, 0]
        k_max = curv[:, 1]
        mean = 0.5 * (k_min + k_max)
        gauss = k_min * k_max
        abs_mean = 0.5 * (np.abs(k_min) + np.abs(k_max))
        abs_max = np.maximum(np.abs(k_min), np.abs(k_max))
        curvedness = np.sqrt(0.5 * (k_min * k_min + k_max * k_max))
        shape = (2.0 / np.pi) * np.arctan2(k_max + k_min, np.maximum(k_max - k_min, 1e-12))
        out[idx, 15] = k_min
        out[idx, 16] = k_max
        out[idx, 17] = mean
        out[idx, 18] = gauss
        out[idx, 19] = abs_mean
        out[idx, 20] = abs_max
        out[idx, 21] = curvedness
        out[idx, 22] = shape

    invalid = ~np.isfinite(out).all(axis=1)
    out[invalid] = 0.0
    return out.astype(np.float32)


def compute_taubin_weingarten_features(
    coords: np.ndarray,
    neighbor_mode: str = "knn",
    k_neighbors: int = 32,
    radius: float = 2.0,
    min_neighbors: int = 10,
    eigenthreshold: float = 1e-5,
    tikhonov: float = 1e-6,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n_points = coords.shape[0]
    out = np.zeros((n_points, TW_FEATURE_DIM), dtype=np.float32)
    if n_points < min_neighbors:
        return out
    tree = cKDTree(coords)
    if neighbor_mode == "knn":
        k = min(max(k_neighbors, min_neighbors), n_points)
        _, indices = tree.query(coords, k=k)
        if indices.ndim == 1:
            indices = indices[:, None]
        centered = coords[indices] - coords[:, None, :]
        return _process_weingarten_batch(centered, eigenthreshold=eigenthreshold, tikhonov=tikhonov)
    if neighbor_mode == "radius":
        neighborhoods = tree.query_ball_point(coords, r=radius)
        for i, idx in enumerate(neighborhoods):
            if len(idx) < min_neighbors:
                continue
            centered = coords[np.asarray(idx)] - coords[i]
            out[i] = _process_weingarten_batch(
                centered[None, :, :],
                eigenthreshold=eigenthreshold,
                tikhonov=tikhonov,
            )[0]
        return out
    raise ValueError(f"Unknown neighbor_mode: {neighbor_mode}")


def compute_tw_lite_features(*args, **kwargs) -> np.ndarray:
    if "ridge_lambda" in kwargs and "tikhonov" not in kwargs:
        kwargs["tikhonov"] = kwargs.pop("ridge_lambda")
    return compute_taubin_weingarten_features(*args, **kwargs)
