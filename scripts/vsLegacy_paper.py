import numpy as np
from pathlib import Path
import argparse
import json
from PIL import Image

# ---------------------------------
# Utilities
# ---------------------------------
def load_rgb_float01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(srgb, 0.0, 1.0)
    a = 0.055
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

# ---------------------------------
# Dual pair split/join (fixed)
# ---------------------------------
# Rec.709 / sRGB luma coefficients (for linear light)
coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
u = np.ones(3, dtype=np.float64) / coeffs.sum()  # ensures <coeffs, u> = 1

def luminance_from(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * coeffs[0] + rgb[..., 1] * coeffs[1] + rgb[..., 2] * coeffs[2]

def split_1p2(rgb: np.ndarray):
    Y = luminance_from(rgb)                          # (H,W)
    r = rgb - Y[..., None] * u[None, None, :]        # (H,W,3)
    return Y, r

def join_1p2(Y: np.ndarray, r: np.ndarray) -> np.ndarray:
    return Y[..., None] * u[None, None, :] + r

# ---------------------------------
# Tone function (1D)
# ---------------------------------
def phi(Y: np.ndarray, alpha: float) -> np.ndarray:
    return np.power(np.clip(Y, 0.0, 1.0), alpha)

# ---------------------------------
# Metrics: C and ΔΣ
# ---------------------------------
def mean_sq_norm(v: np.ndarray) -> float:
    vv = v.reshape(-1, 3)
    return float(np.mean(np.sum(vv * vv, axis=1)))

def cov_centered(r: np.ndarray) -> np.ndarray:
    X = r.reshape(-1, 3)
    mu = np.mean(X, axis=0, keepdims=True)
    Z = X - mu
    return (Z.T @ Z) / X.shape[0]

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-minimal image experiment: compare CONJ vs legacy using C and ΔΣ.")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", type=str, default=str(repo_root / "assets" / "images" / "YKT_3336.jpg"))
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--outdir", type=str, default=str(repo_root / "output"))
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    rgb_srgb = load_rgb_float01(in_path)

    # ---- CONJ pipeline: linear -> split -> phi(Y) -> join ----
    rgb_lin = srgb_to_linear(rgb_srgb)
    Y_lin, r_lin = split_1p2(rgb_lin)
    Y_lin_t = phi(Y_lin, args.alpha)
    rgb_conj_lin = join_1p2(Y_lin_t, r_lin)

    # ---- Legacy pipeline: gamma(sRGB) -> split -> phi(Y) -> join ----
    # (Legacy here means splitting in a non-linear (gamma-encoded) space.)
    Y_gam, r_gam = split_1p2(rgb_srgb)
    Y_gam_t = phi(Y_gam, args.alpha)
    rgb_legacy_srgb = np.clip(join_1p2(Y_gam_t, r_gam), 0.0, 1.0)
    rgb_legacy_lin = srgb_to_linear(rgb_legacy_srgb)

    # ---- Metrics in linear space (consistent with the 1+2 definition) ----
    r_orig = split_1p2(rgb_lin)[1]
    r_conj = split_1p2(rgb_conj_lin)[1]
    r_legacy = split_1p2(rgb_legacy_lin)[1]

    C_conj = mean_sq_norm(r_conj - r_orig)
    C_legacy = mean_sq_norm(r_legacy - r_orig)

    S_orig = cov_centered(r_orig)
    S_conj = cov_centered(r_conj)
    S_legacy = cov_centered(r_legacy)

    DeltaSigma_conj = float(np.linalg.norm(S_conj - S_orig, ord="fro"))
    DeltaSigma_legacy = float(np.linalg.norm(S_legacy - S_orig, ord="fro"))

    result = {
        "input": str(in_path),
        "alpha": float(args.alpha),
        "C_conj": C_conj,
        "C_legacy": C_legacy,
        "DeltaSigma_conj_Fro": DeltaSigma_conj,
        "DeltaSigma_legacy_Fro": DeltaSigma_legacy,
    }

    stem = in_path.stem
    with open(outdir / f"{stem}_paper_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
