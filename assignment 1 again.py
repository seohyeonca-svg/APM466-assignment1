from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# =========================
# INPUTS
# =========================
FACE = 100.0

PRICING_DATES = np.array(pd.to_datetime([
    "2026-01-05","2026-01-06","2026-01-07","2026-01-08","2026-01-09",
    "2026-01-12","2026-01-13","2026-01-14","2026-01-15","2026-01-16",
]).date)

MATURITIES = np.array(pd.to_datetime([
    "2026-03-01","2026-09-01",
    "2027-03-01","2027-09-01",
    "2028-03-01","2028-09-01",
    "2029-03-01","2029-09-01",
    "2030-03-01","2030-09-01",
]).date)

# Replace with your actual coupon rates (annual coupon as decimal)
COUPON_RATES = np.array([0.0025,0.0100,0.0125,0.0275,0.0350,0.0325,0.0400,0.0350,0.0275,0.0275], float)

# Clean prices (10 bonds x 10 days)
CLEAN_PRICE_LISTS = [
    [99.70, 99.71, 99.71, 99.72, 99.73, 99.74, 99.74, 99.75, 99.76, 99.77],
    [99.14, 99.14, 99.17, 99.16, 99.19, 99.18, 99.19, 99.20, 99.21, 99.22],
    [98.60, 98.63, 98.66, 98.67, 98.67, 98.67, 98.68, 98.67, 98.73, 98.72],
    [100.22, 100.30, 100.28,100.31,100.30,100.32,100.30,100.31,100.35,100.37],
    [101.73,101.78,101.78,101.80,101.79,101.81,101.78,101.81,101.84,101.83],
    [101.34,101.41,101.40,101.43,101.42,101.45,101.42,101.43,101.48,101.47],
    [103.63,103.70,103.71,103.74,103.73,103.76,103.72,103.76,103.79,103.78],
    [102.22,102.33,102.37,102.34,102.31,102.35,102.29,102.33,102.43,102.42],
    [99.49, 99.42, 99.56, 99.50, 99.58, 99.53, 99.50, 99.66, 99.66, 99.61],
    [99.16, 99.08, 99.25, 99.17, 99.25, 99.21, 99.19, 99.36, 99.36, 99.31],
]

clean_prices = np.column_stack([np.array(lst, float) for lst in CLEAN_PRICE_LISTS])
if clean_prices.shape != (10, 10):
    raise ValueError(f"Expected (10,10), got {clean_prices.shape}")

# rounding used in bootstrap times to avoid float mismatch
TIME_ROUND = 10

# =========================
# HELPERS
# =========================

def cashflow_semiannual(maturity: date, settle: date):
    """Semiannual coupon dates AFTER settle up to maturity, stepping by 6 months."""
    d = maturity
    while d > settle:
        d -= relativedelta(months=6)
    d += relativedelta(months=6)

    dates = []
    while d <= maturity:
        dates.append(d)
        d += relativedelta(months=6)
    return dates


def dirty_prices_from_clean(maturity_date: date, cpn_rate: float,
                            clean_vec: np.ndarray, pricing_dates: np.ndarray,
                            face: float):
    """
    AI = (n/365) * (face * coupon_rate)
    Dirty = Clean + AI
    """
    out = []
    for settle, clean in zip(pricing_dates, clean_vec):
        last_cpn = maturity_date
        while last_cpn > settle:
            last_cpn -= relativedelta(months=6)
        n = (settle - last_cpn).days
        AI = (face * cpn_rate * n) / 365.0
        out.append(clean + AI)
    return np.array(out, float)


def pv_from_y_cont(y, settle, maturity, cpn_rate, face):
    """P = sum CF * exp(-y t), t=days/365, semiannual coupons, continuous comp."""
    dates = cashflow_semiannual(maturity, settle)
    if len(dates) == 0:
        return face
    coupon = face * cpn_rate / 2.0
    pv = 0.0
    for d in dates:
        t = (d - settle).days / 365.0
        cf = coupon + (face if d == maturity else 0.0)
        pv += cf * np.exp(-y * t)
    return pv


def solve_ytm_continuous(dirty_price, settle, maturity, cpn_rate, face):
    """Solve for y in continuous-compounding bond pricing equation."""
    f = lambda y: pv_from_y_cont(y, settle, maturity, cpn_rate, face) - dirty_price
    lo, hi = -0.50, 2.00
    if f(lo) * f(hi) > 0:
        raise RuntimeError(f"Cannot bracket YTM root for settle={settle}, maturity={maturity}")
    return brentq(f, lo, hi, xtol=1e-12, maxiter=200)

def linear_interp(x, y, extrapolate=False):
    """Linear interpolation helper. If extrapolate=False, out-of-range -> NaN."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    fill = "extrapolate" if extrapolate else np.nan
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value=fill)

# =========================
# (a) YTM: compute + plot 0–5y curves (superimposed)
# =========================
dirty_prices = np.zeros((10,10), float)
ytm = np.zeros((10,10), float)
ttm = np.zeros((10,10), float)

# Dirty prices by bond/day
for j in range(10):
    dirty_prices[:, j] = dirty_prices_from_clean(
        MATURITIES[j], float(COUPON_RATES[j]), clean_prices[:, j], PRICING_DATES, FACE
    )

# Compute time-to-maturity (days/365) and YTM (continuous)
for i in range(10):
    settle = PRICING_DATES[i]
    for j in range(10):
        maturity = MATURITIES[j]
        ttm[i, j] = (maturity - settle).days / 365.0
        ytm[i, j] = solve_ytm_continuous(dirty_prices[i, j], settle, maturity, COUPON_RATES[j], FACE)

ytm_df = pd.DataFrame(
    ytm,
    index=pd.to_datetime(PRICING_DATES),
    columns=[f"Bond{j+1}@{MATURITIES[j]}" for j in range(10)]
)
print("\n(a) Continuously-compounded YTMs (decimal):")
print(ytm_df.round(10))

# Plot YTM curves on a 0–5y grid
T_grid = np.linspace(0.0, 5.0, 301)
plt.figure(figsize=(10,6))

for i, settle_ts in enumerate(pd.to_datetime(PRICING_DATES)):
    T = ttm[i,:]
    Y = ytm[i,:]
    m = (T > 0) & (T <= 5.0) & np.isfinite(Y)
    if np.sum(m) < 2:
        continue

    interp = linear_interp(T[m], Y[m], extrapolate=False)
    Yg = interp(T_grid)
    mask = np.isfinite(Yg)  # avoid plotting NaNs outside interpolation range
    plt.plot(T_grid[mask], 100*Yg[mask], linewidth=1.5, label=str(settle_ts.date()))

plt.xlabel("Time to maturity (years, days/365)")
plt.ylabel("YTM (% per year, continuous comp)")
plt.title("5-Year YTM Curves (10 days, superimposed) — Linear interpolation")
plt.grid(True, alpha=0.3)
plt.xlim(0,5)
plt.legend(title="Pricing date", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

# =========================
# (b) BOOTSTRAP spot curve (continuous comp) + plot 0–5y
# =========================
def bootstrap_spot_curve_for_day(settle: date, maturities, coupon_rates, dirty_prices_day, face,
                                 time_round=10):

    order = np.argsort(maturities)
    maturities = maturities[order]
    coupon_rates = coupon_rates[order]
    dirty_prices_day = dirty_prices_day[order]

    df_map = {}  # key: rounded t, value: DF(t)

    for maturity, cpn, price in zip(maturities, coupon_rates, dirty_prices_day):
        dates = cashflow_semiannual(maturity, settle)
        if len(dates) == 0:
            continue

        coupon = face * cpn / 2.0
        times = [round((d - settle).days/365.0, time_round) for d in dates]
        cfs = [coupon + (face if d == maturity else 0.0) for d in dates]

        pv_known = 0.0
        for t, cf in zip(times[:-1], cfs[:-1]):
            if t not in df_map:
                raise RuntimeError(
                    f"Missing DF for t={t} when bootstrapping maturity={maturity} on settle={settle}. "
                    "This means your cashflow times don't line up in increasing order."
                )
            pv_known += cf * df_map[t]

        T_last = times[-1]
        CF_last = cfs[-1]
        DF_last = (price - pv_known) / CF_last

        if DF_last <= 0:
            raise RuntimeError(f"DF<=0 at maturity={maturity}, settle={settle}. Check inputs.")
        df_map[T_last] = DF_last

    t_pts = np.array(sorted(df_map.keys()), float)
    df_pts = np.array([df_map[t] for t in t_pts], float)
    r_pts = -np.log(df_pts) / t_pts
    return t_pts, r_pts

# we do bootstrap_spot_curve_for_day for jan5, ..., jan 16
spot_rates_by_day = []
for i in range(10):
    t_pts, r_pts = bootstrap_spot_curve_for_day(
        PRICING_DATES[i], MATURITIES, COUPON_RATES, dirty_prices[i,:], FACE, time_round=TIME_ROUND
    )
    spot_rates_by_day.append((t_pts, r_pts))


spot_at_maturities = np.full((10,10), np.nan, float)
for i in range(10):
    Tm = np.round(ttm[i,:], TIME_ROUND)
    t_pts, r_pts = spot_rates_by_day[i]
    interp = linear_interp(t_pts, r_pts, extrapolate=False)

    tmin, tmax = np.min(t_pts), np.max(t_pts)
    Tm_clip = np.clip(Tm, tmin, tmax)

    spot_at_maturities[i,:] = interp(Tm_clip)

spot_df = pd.DataFrame(
    spot_at_maturities,
    index=pd.to_datetime(PRICING_DATES),
    columns=[f"Spot@{MATURITIES[j]}" for j in range(10)]
)
print("\n(b) Bootstrapped spot rates at each bond maturity (decimal, continuous comp):")
print(spot_df.round(10))

# Plot 0–5y spot curves (superimposed)
plt.figure(figsize=(10,6))
for i, settle_ts in enumerate(pd.to_datetime(PRICING_DATES)):
    t_pts, r_pts = spot_rates_by_day[i]
    m = (t_pts > 0) & (t_pts <= 5.0) & np.isfinite(r_pts)
    if np.sum(m) < 2:
        continue

    interp = linear_interp(t_pts[m], r_pts[m], extrapolate=False)
    Rg = interp(T_grid)
    mask = np.isfinite(Rg)
    plt.plot(T_grid[mask], 100*Rg[mask], linewidth=1.5, label=str(settle_ts.date()))

plt.xlabel("Time (years, days/365)")
plt.ylabel("Spot rate r(t) (% per year, continuous comp)")
plt.title("5-Year Spot Curves (Bootstrapped, 10 days, superimposed) — Linear interpolation")
plt.grid(True, alpha=0.3)
plt.xlim(0,5)
plt.legend(title="Pricing date", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

# =========================
# (c) 1-YEAR FORWARD CURVE from spot curve (continuous comp)
# =========================
spot_knots = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
forward_1y = np.full((len(PRICING_DATES), 4), np.nan, dtype=float)

for i in range(len(PRICING_DATES)):
    t_pts, r_pts = spot_rates_by_day[i]

    # linear interpolation for S(t); allow extrapolation here so S(5) exists even if last knot < 5
    spot_interp = linear_interp(t_pts, r_pts, extrapolate=True)
    S1, S2, S3, S4, S5 = map(float, spot_interp(spot_knots))

    forward_1y[i, 0] = 2.0*S2 - 1.0*S1
    forward_1y[i, 1] = 3.0*S3 - 2.0*S2
    forward_1y[i, 2] = 4.0*S4 - 3.0*S3
    forward_1y[i, 3] = 5.0*S5 - 4.0*S4

forward_df = pd.DataFrame(
    forward_1y,
    index=pd.to_datetime(PRICING_DATES),
    columns=["F(1,2)", "F(2,3)", "F(3,4)", "F(4,5)"]
)
print("\n(c) 1-year forward rates (continuous comp, decimal):")
print(forward_df.round(10))

# Plot forward curves (segments)
plt.figure(figsize=(10, 6))
x_labels = ["1y-1y", "1y-2y", "1y-3y", "1y-4y"]
x = np.arange(len(x_labels))

for i, settle_ts in enumerate(pd.to_datetime(PRICING_DATES)):
    y = forward_1y[i, :]
    if np.any(~np.isfinite(y)):
        continue
    plt.plot(x, 100*y, marker="o", linewidth=1.5, label=str(settle_ts.date()))

plt.xticks(x, x_labels)
plt.xlabel("1-year forward segment")
plt.ylabel("Forward rate (% per year, continuous comp)")
plt.title("1-Year Forward Curves (10 days superimposed) — Linear interpolation")
plt.grid(True, alpha=0.3)
plt.legend(title="Pricing date", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

ytm_df.round(10).to_csv("q2_ytm_table.csv")
spot_df.round(10).to_csv("q2_spot_table.csv")
forward_df.round(10).to_csv("q2_forward_table.csv")

# ============================================================
# Q5 (Covariance matrices)
# ============================================================

# YIELD PART
YIELD_MATS = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)  # i = 1,...,5

yields_daily = np.full((len(PRICING_DATES), len(YIELD_MATS)), np.nan, dtype=float)

for j_day in range(len(PRICING_DATES)):

    T_pts = ttm[j_day, :]
    Y_pts = ytm[j_day, :]

    good = (T_pts > 0) & np.isfinite(Y_pts)
    if np.sum(good) < 2:
        raise RuntimeError(f"Not enough YTM points to build yield curve on {PRICING_DATES[j_day]}")

    # linear interpolation of the YTM curve; allow extrapolation to evaluate at 5Y if needed
    y_curve = linear_interp(T_pts[good], Y_pts[good], extrapolate=True)

    yields_daily[j_day, :] = y_curve(YIELD_MATS)

yields_daily_df = pd.DataFrame(
    yields_daily,
    index=pd.to_datetime(PRICING_DATES),
    columns=[f"Y_{int(m)}Y" for m in YIELD_MATS]
)

print("\nQ5: r_{i,j} yield levels from Part (a) YTM curve (10 days x 5 maturities), decimals:")
print(yields_daily_df.round(10))

# Compute X_{i,j} = log(r_{i,j+1} / r_{i,j}), j=1,...,9
if np.any(~np.isfinite(yields_daily)):
    raise RuntimeError("Non-finite yield level encountered in yields_daily; cannot take log-returns.")
if np.any(yields_daily <= 0):
    raise RuntimeError(
        "Non-positive yield encountered in yields_daily; log(r_{t+1}/r_t) requires positive rates."
    )

X_yields = np.log(yields_daily[1:, :] / yields_daily[:-1, :])

X_yields_df = pd.DataFrame(
    X_yields,
    index=pd.to_datetime(PRICING_DATES[1:]),  # date of the later day in the ratio
    columns=[f"X_logret_Y_{int(m)}Y" for m in YIELD_MATS]
)

print("\nQ5: X_{i,j} = daily log-returns of yields (9 x 5):")
print(X_yields_df.round(10))

# Covariance matrix of yield log-returns (5x5)
cov_yields = np.cov(X_yields, rowvar=False, ddof=1)
cov_yields_df = pd.DataFrame(
    cov_yields,
    index=[f"Y_{int(m)}Y" for m in YIELD_MATS],
    columns=[f"Y_{int(m)}Y" for m in YIELD_MATS]
)

print("\nQ5: Covariance matrix of daily log-returns of yields (5 x 5):")
print(cov_yields_df.round(12))


# FORWARDS PART
F = forward_1y  # shape (10 x 4)
F_NAMES = ["1yr-1yr", "1yr-2yr", "1yr-3yr", "1yr-4yr"]

F_df = pd.DataFrame(F, index=pd.to_datetime(PRICING_DATES), columns=F_NAMES)
print("\nQ5: r_{i,j} forward levels from Part (c) (10 days x 4), decimals:")
print(F_df.round(10))

if np.any(~np.isfinite(F)):
    raise RuntimeError("Non-finite forward level encountered; cannot take log-returns.")
if np.any(F <= 0):
    raise RuntimeError(
        "Non-positive forward encountered; log(r_{t+1}/r_t) requires positive rates."
    )

X_fwds = np.log(F[1:, :] / F[:-1, :])  # shape (9 x 4)

X_fwds_df = pd.DataFrame(
    X_fwds,
    index=pd.to_datetime(PRICING_DATES[1:]),
    columns=[f"X_logret_{name}" for name in F_NAMES]
)

print("\nQ5: X_{i,j} = daily log-returns of forward rates (9 x 4):")
print(X_fwds_df.round(10))

cov_fwds = np.cov(X_fwds, rowvar=False, ddof=1)
cov_fwds_df = pd.DataFrame(cov_fwds, index=F_NAMES, columns=F_NAMES)

print("\nQ5: Covariance matrix of daily log-returns of forward rates (4 x 4):")
print(cov_fwds_df.round(12))

# Save to CSV
yields_daily_df.round(10).to_csv("q5_yield_levels_5mats.csv")
X_yields_df.round(10).to_csv("q5_yield_logreturns.csv")
cov_yields_df.round(12).to_csv("q5_cov_yield_logreturns.csv")

F_df.round(10).to_csv("q5_forward_levels.csv")
X_fwds_df.round(10).to_csv("q5_forward_logreturns.csv")
cov_fwds_df.round(12).to_csv("q5_cov_forward_logreturns.csv")

# ============================================================
# Q6 Eigen-decomposition of covariance matrices
# ============================================================

import numpy as np
import pandas as pd

# =========================
# 1) YIELD covariance matrix (5 x 5)
# =========================
Sigma_y = cov_yields_df.values

# eigenvalues & eigenvectors
eigvals_y, eigvecs_y = np.linalg.eigh(Sigma_y)

# sort in descending order (largest eigenvalue first)
idx = np.argsort(eigvals_y)[::-1]
eigvals_y = eigvals_y[idx]
eigvecs_y = eigvecs_y[:, idx]

eigvals_y_df = pd.DataFrame(
    eigvals_y,
    index=[f"PC{i+1}" for i in range(len(eigvals_y))],
    columns=["Eigenvalue"]
)

eigvecs_y_df = pd.DataFrame(
    eigvecs_y,
    index=cov_yields_df.index,
    columns=[f"PC{i+1}" for i in range(len(eigvals_y))]
)

print("\nEigenvalues of yield log-return covariance matrix:")
print(eigvals_y_df.round(12))

print("\nEigenvectors of yield log-return covariance matrix (columns = PCs):")
print(eigvecs_y_df.round(6))

# =========================
# 2) FORWARD covariance matrix (4 x 4)
# =========================
Sigma_f = cov_fwds_df.values

eigvals_f, eigvecs_f = np.linalg.eigh(Sigma_f)

# sort descending
idx = np.argsort(eigvals_f)[::-1]
eigvals_f = eigvals_f[idx]
eigvecs_f = eigvecs_f[:, idx]

eigvals_f_df = pd.DataFrame(
    eigvals_f,
    index=[f"PC{i+1}" for i in range(len(eigvals_f))],
    columns=["Eigenvalue"]
)

eigvecs_f_df = pd.DataFrame(
    eigvecs_f,
    index=cov_fwds_df.index,
    columns=[f"PC{i+1}" for i in range(len(eigvals_f))]
)

print("\nEigenvalues of forward log-return covariance matrix:")
print(eigvals_f_df.round(12))

print("\nEigenvectors of forward log-return covariance matrix (columns = PCs):")
print(eigvecs_f_df.round(6))

# Save results
eigvals_y_df.to_csv("q5_yield_eigenvalues.csv")
eigvecs_y_df.to_csv("q5_yield_eigenvectors.csv")

eigvals_f_df.to_csv("q5_forward_eigenvalues.csv")
eigvecs_f_df.to_csv("q5_forward_eigenvectors.csv")
