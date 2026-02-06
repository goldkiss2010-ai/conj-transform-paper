import numpy as np

# ============================================
# 基本設定
# ============================================

N = 100000          # サンプル数（必要に応じて変更）
RNG_SEED = 12345   # 乱数シード

rng = np.random.default_rng(RNG_SEED)

# scene-linear sRGB を一様サンプリング
sRGB_lin = rng.random((N, 3))  # N x 3, 各成分 [0,1]

# sRGB (D65) linear RGB -> XYZ 行列
M_srgb_to_xyz = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

M_xyz_to_srgb = np.linalg.inv(M_srgb_to_xyz)

# XYZ サンプル
XYZ = (M_srgb_to_xyz @ sRGB_lin.T).T  # N x 3

# 数値的にごくわずかな負値をクリップ
sRGB_lin = np.clip(sRGB_lin, 0.0, 1.0)
XYZ = np.clip(XYZ, 0.0, None)


# ============================================
# 双対ペアと射影
# ============================================

# XYZ 側の dual pair (u_Y, ℓ_Y)
u_Y = np.array([0.0, 1.0, 0.0])

def ell_Y(x_xyz: np.ndarray) -> float:
    return float(x_xyz[1])  # Y 成分


# sRGB 側の dual pair (u_s, ℓ_s)
u_s = M_xyz_to_srgb @ u_Y

def ell_s(x_srgb_lin: np.ndarray) -> float:
    xyz = M_srgb_to_xyz @ x_srgb_lin
    return float(xyz[1])  # Y

def project_kernel(x: np.ndarray, u: np.ndarray, ell) -> np.ndarray:
    """P_{ker ℓ}(x) = x - ℓ(x) u"""
    s = ell(x)
    return x - s * u


# ============================================
# CONJ 演算子
# ============================================

def conj_operator(x: np.ndarray, u: np.ndarray, ell, phi) -> np.ndarray:
    s = ell(x)
    return phi(s) * u + (x - s * u)


def phi_power(s: float, alpha: float = 0.8) -> float:
    """単調なべき関数 phi(s) = s^alpha （s>=0 前提）"""
    s_clipped = max(s, 0.0)
    return s_clipped ** alpha


# ============================================
# 指標 C(F;D) とクロマ共分散
# ============================================

def crosstalk(F, D: np.ndarray, u: np.ndarray, ell) -> float:
    diffs = []
    for x in D:
        x = np.asarray(x, dtype=float)
        y = F(x)
        r_x = project_kernel(x, u, ell)
        r_y = project_kernel(y, u, ell)
        diffs.append(np.linalg.norm(r_y - r_x) ** 2)
    return float(np.mean(diffs))


def chroma_cov(D: np.ndarray, u: np.ndarray, ell) -> np.ndarray:
    R = np.array([project_kernel(x, u, ell) for x in D])
    R_mean = np.mean(R, axis=0)
    C = R - R_mean
    return (C.T @ C) / len(D)


def frob_norm(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))


# ============================================
# sRGB 側：従来パイプライン（簡易 Y'CbCr モデル）
# ============================================

def gamma_encode(x_lin: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    x = np.clip(x_lin, 0.0, None)
    return x ** (1.0 / gamma)


def gamma_decode(x_nonlin: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    x = np.clip(x_nonlin, 0.0, None)
    return x ** gamma


def rgb_to_ycbcr_bt709(rgb_nl: np.ndarray) -> np.ndarray:
    """
    簡易 BT.709 Y'CbCr 変換（オフセットなしの正規化バージョン）。
    """
    r, g, b = rgb_nl
    y  = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    return np.array([y, cb, cr])


def ycbcr_to_rgb_bt709(ycbcr: np.ndarray) -> np.ndarray:
    """
    簡易 BT.709 Y'CbCr 逆変換（オフセットなしの正規化バージョン）。
    """
    y, cb, cr = ycbcr
    r = cr * 1.5748 + y
    b = cb * 1.8556 + y
    # g は Y' = 0.2126 R' + 0.7152 G' + 0.0722 B' から復元
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.array([r, g, b])


def tone_curve_y(y: float, alpha: float = 0.8) -> float:
    """Y' に対するトーンカーブ（単純なべき）"""
    y_clipped = np.clip(y, 0.0, 1.0)
    return y_clipped ** alpha


def F_legacy_srgb(x_lin: np.ndarray,
                  gamma: float = 2.2,
                  alpha: float = 0.8) -> np.ndarray:
    """
    scene-linear sRGB -> ガンマ -> Y'CbCr -> Y' トーン -> Y'CbCr -> ガンマ逆
    """
    # gamma encode
    x_nl = gamma_encode(x_lin, gamma)
    # Y'CbCr
    ycbcr = rgb_to_ycbcr_bt709(x_nl)
    # Y' のみトーンカーブ
    ycbcr[0] = tone_curve_y(ycbcr[0], alpha=alpha)
    # 逆変換
    x_nl_new = ycbcr_to_rgb_bt709(ycbcr)
    # gamma decode して scene-linear に戻す
    x_lin_new = gamma_decode(x_nl_new, gamma)
    return x_lin_new


def F_conj_srgb(x_lin: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """
    scene-linear sRGB 上の CONJ。
    φ(s) = s^alpha とする。
    """
    return conj_operator(x_lin, u_s, ell_s,
                         lambda s: phi_power(s, alpha=alpha))


# ============================================
# CIELAB 用 XYZ <-> Lab 変換
# ============================================

# D65 白色点（Y=1 正規化）
WHITE_XYZ = np.array([0.95047, 1.00000, 1.08883])

EPS = 1e-12
DELTA = 6 / 29
DELTA3 = DELTA ** 3

def f_lab(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.where(t > DELTA3, np.cbrt(np.clip(t, 0.0, None)),
                    t / (3 * DELTA ** 2) + 4 / 29)


def finv_lab(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return np.where(f > DELTA, f ** 3,
                    3 * DELTA ** 2 * (f - 4 / 29))


def xyz_to_lab(xyz: np.ndarray,
               white_xyz: np.ndarray = WHITE_XYZ) -> np.ndarray:
    X, Y, Z = xyz
    Xn, Yn, Zn = white_xyz
    xr = X / (Xn + EPS)
    yr = Y / (Yn + EPS)
    zr = Z / (Zn + EPS)

    fx = f_lab(xr)
    fy = f_lab(yr)
    fz = f_lab(zr)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b])


def lab_to_xyz(lab: np.ndarray,
               white_xyz: np.ndarray = WHITE_XYZ) -> np.ndarray:
    L, a, b = lab
    Xn, Yn, Zn = white_xyz

    fy = (L + 16) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xr = finv_lab(fx)
    yr = finv_lab(fy)
    zr = finv_lab(fz)

    X = xr * Xn
    Y = yr * Yn
    Z = zr * Zn
    return np.array([X, Y, Z])


def tone_curve_L(L: float, alpha: float = 0.8) -> float:
    """L* に対するトーンカーブ。0–100 を 0–1 に正規化してべき。"""
    L_norm = np.clip(L / 100.0, 0.0, 1.0)
    L_new = (L_norm ** alpha) * 100.0
    return float(L_new)


def F_Lab_legacy_xyz(x_xyz: np.ndarray,
                     alpha: float = 0.8) -> np.ndarray:
    """
    XYZ -> Lab -> L* トーン -> Lab -> XYZ
    """
    lab = xyz_to_lab(x_xyz, WHITE_XYZ)
    L, a, b = lab
    L_new = tone_curve_L(L, alpha=alpha)
    lab_new = np.array([L_new, a, b])
    xyz_new = lab_to_xyz(lab_new, WHITE_XYZ)
    return xyz_new


def F_XYZ_conj(x_xyz: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """
    XYZ 空間上の CONJ。φ(s) = s^alpha とする。
    """
    return conj_operator(x_xyz, u_Y, ell_Y,
                         lambda s: phi_power(s, alpha=alpha))


# ============================================
# 実験実行関数
# ============================================

def run_experiment_srgb(alpha: float = 0.8, gamma: float = 2.2):
    D = sRGB_lin  # N x 3
    Sigma_base = chroma_cov(D, u_s, ell_s)

    # 従来パイプライン
    def F_leg(x):
        return F_legacy_srgb(x, gamma=gamma, alpha=alpha)

    D_legacy = np.array([F_leg(x) for x in D])
    C_legacy = crosstalk(F_leg, D, u_s, ell_s)
    Sigma_legacy = chroma_cov(D_legacy, u_s, ell_s)

    # CONJ
    def F_conj_loc(x):
        return F_conj_srgb(x, alpha=alpha)

    D_conj = np.array([F_conj_loc(x) for x in D])
    C_conj = crosstalk(F_conj_loc, D, u_s, ell_s)
    Sigma_conj = chroma_cov(D_conj, u_s, ell_s)

    print("=== sRGB-side experiment (uniform samples) ===")
    print(f"N = {len(D)}  samples, alpha = {alpha}, gamma = {gamma}")
    print(f"C_legacy = {C_legacy:.6e}")
    print(f"C_conj   = {C_conj:.6e}")
    print("ΔSigma_legacy (Frobenius) =",
          f"{frob_norm(Sigma_legacy - Sigma_base):.6e}")
    print("ΔSigma_conj   (Frobenius) =",
          f"{frob_norm(Sigma_conj   - Sigma_base):.6e}")
    print()


def run_experiment_lab(alpha: float = 0.8):
    D = XYZ  # N x 3
    Sigma_base = chroma_cov(D, u_Y, ell_Y)

    # Lab-based pipeline
    def F_leg(x):
        return F_Lab_legacy_xyz(x, alpha=alpha)

    D_legacy = np.array([F_leg(x) for x in D])
    C_legacy = crosstalk(F_leg, D, u_Y, ell_Y)
    Sigma_legacy = chroma_cov(D_legacy, u_Y, ell_Y)

    # XYZ CONJ
    def F_conj_loc(x):
        return F_XYZ_conj(x, alpha=alpha)

    D_conj = np.array([F_conj_loc(x) for x in D])
    C_conj = crosstalk(F_conj_loc, D, u_Y, ell_Y)
    Sigma_conj = chroma_cov(D_conj, u_Y, ell_Y)

    print("=== XYZ/Lab-side experiment (uniform samples) ===")
    print(f"N = {len(D)}  samples, alpha = {alpha}")
    print(f"C_legacy = {C_legacy:.6e}")
    print(f"C_conj   = {C_conj:.6e}")
    print("ΔSigma_legacy (Frobenius) =",
          f"{frob_norm(Sigma_legacy - Sigma_base):.6e}")
    print("ΔSigma_conj   (Frobenius) =",
          f"{frob_norm(Sigma_conj   - Sigma_base):.6e}")
    print()


if __name__ == "__main__":
    # sRGB 側（ガンマ後 Y'CbCr で Luma 処理） vs CONJ
    run_experiment_srgb(alpha=0.8, gamma=2.2)

    # XYZ/Lab 側（Lab で L* トーンカーブ） vs XYZ 上の CONJ
    run_experiment_lab(alpha=0.8)
